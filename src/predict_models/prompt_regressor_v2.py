# prompt_angle_regressor_advanced.py
import math

import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_is_fitted

from src.predict_models.garbage_detector import GarbageDetector


# -------------------------
# Helpers: principal angles + spectral features
# -------------------------
def principal_singulars(Ua: np.ndarray, Ub: np.ndarray) -> np.ndarray:
    """
    Compute singular values (sigma) between two orthonormal bases Ua (d x k1) and Ub (d x k2).
    Returns sigma sorted decreasingly.
    """
    if Ua is None or Ub is None or Ua.size == 0 or Ub.size == 0:
        return np.array([], dtype=float)

    C = Ua.T @ Ub  # (k1 x k2)
    s = np.linalg.svd(C, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return np.sort(s)[::-1]


def spectral_features_from_sigma(s: np.ndarray) -> Dict[str, float]:
    """
    Given array of singular values s (descending), compute many spectral descriptors.
    """
    feats: Dict[str, float] = {}
    if s.size == 0:
        # zeros for degenerate case
        zeros = {
            "sigma_max": 0.0, "sigma_min": 0.0, "sigma_mean": 0.0, "sigma_std": 0.0,
            "sigma_energy": 0.0, "sigma_entropy": 0.0, "sigma_range": 0.0,
            "sigma_norm1": 0.0, "sigma_norm2": 0.0, "sigma_frac_gt_08": 0.0,
            "sigma_frac_gt_05": 0.0, "spectral_flatness": 0.0, "spectral_sharpness": 0.0
        }
        feats.update(zeros)
        return feats

    s_max = float(np.max(s))
    s_min = float(np.min(s))
    s_mean = float(np.mean(s))
    s_std = float(np.std(s))
    energy = float(np.sum(s ** 2))
    norm1 = float(np.sum(s))
    norm2 = float(np.linalg.norm(s))
    s_range = s_max - s_min
    frac_gt_08 = float(np.mean(s > 0.8))
    frac_gt_05 = float(np.mean(s > 0.5))

    # entropy: treat s normalized to sum 1 (add eps)
    eps = 1e-12
    p = s / (np.sum(s) + eps)
    entropy = float(-np.sum(np.where(p > 0, p * np.log(p + eps), 0.0)))

    # spectral_flatness: geom_mean / arith_mean
    try:
        geom_mean = float(np.exp(np.mean(np.log(s + eps))))
        arith_mean = s_mean + eps
        flatness = geom_mean / arith_mean
    except Exception:
        flatness = 0.0

    # spectral_sharpness: sigma1 / mean(rest) (guard)
    if s.size > 1:
        sharpness = float(s_max / (np.mean(s[1:]) + eps))
    else:
        sharpness = float(s_max / (s_mean + eps))

    # slope / curvature as simple proxies
    slope = float((s[0] - s[-1]) / max(1, len(s)))
    curve = float(np.sum((s - s_mean) ** 2))

    feats.update({
        "sigma_max": s_max,
        "sigma_min": s_min,
        "sigma_mean": s_mean,
        "sigma_std": s_std,
        "sigma_energy": energy,
        "sigma_entropy": entropy,
        "sigma_range": s_range,
        "sigma_norm1": norm1,
        "sigma_norm2": norm2,
        "sigma_frac_gt_08": frac_gt_08,
        "sigma_frac_gt_05": frac_gt_05,
        "spectral_flatness": flatness,
        "spectral_sharpness": sharpness,
        "sigma_slope": slope,
        "sigma_curve": curve
    })
    return feats


def angle_basic_from_sigma(s: np.ndarray) -> Dict[str, float]:
    """
    Convert singular values -> angles (for each sigma: angle = arccos(sigma)).
    Provide primary-angle-derived features (for the principal singular).
    """
    feats: Dict[str, float] = {}
    if s.size == 0:
        return {
            "angle_rad": 0.0,
            "angle_deg": 0.0,
            "angle_norm": 0.0,
            "sin_angle": 0.0,
            "cos_angle": 1.0,
            "misalignment": 1.0
        }
    # principal sigma is first (largest)
    s1 = float(s[0])
    s1 = max(min(s1, 1.0), -1.0)
    angle = float(np.arccos(s1))
    angle_deg = float(np.degrees(angle))
    angle_norm = angle / math.pi
    sin_angle = float(math.sin(angle))
    cos_angle = s1
    misalignment = 1.0 - s1
    return {
        "angle_rad": angle,
        "angle_deg": angle_deg,
        "angle_norm": angle_norm,
        "sin_angle": sin_angle,
        "cos_angle": cos_angle,
        "misalignment": misalignment
    }


# -------------------------
# The advanced sklearn-like regressor
# -------------------------
class PromptAngleRegressorAdvanced(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    Advanced prompt-aware linear validator.
    - prompt: anchor prompt text
    - n_components: SVD embedding dim
    - answer_subspace_dim: k for local answer subspace (>=1)
    - model_alpha: Ridge regularization strength
    - random_state: for reproducibility in small-noise subspace generation
    """
    name: str = "PromptAngleRegressor"

    tfidf_: TfidfVectorizer = None
    svd_: TruncatedSVD = None
    model_: Ridge = None

    garbage_detector: GarbageDetector = None

    def __init__(
            self,
            prompt: str,
            n_components: int = 100,
            answer_subspace_dim: int = 1,
            model_alpha: float = 1.0,
            stop_words: Optional[str] = "russian",
            random_state: int = 42,
            garbage_penalty: float = 0.0
    ):
        self.prompt = prompt
        self.n_components = n_components
        self.answer_subspace_dim = int(answer_subspace_dim)
        self.model_alpha = model_alpha
        self.stop_words = stop_words
        self.random_state = int(random_state)
        self.garbage_penalty = garbage_penalty

        # Will be created at fit time:
        self.U_prompt_ = None
        self.angle_feature_names_ = []  # recorded order

    # -------------------------
    # Build local subspace for a vector (embedding)
    # -------------------------
    def _make_local_subspace(self, vec: np.ndarray, k: int) -> np.ndarray:
        """
        vec: 1D vector length d (embedding)
        k: desired subspace dimension (>=1)
        Returns Ua: shape (d, k) orthonormal columns.
        Implementation: if k==1 => normalized vec; else build tiny noise samples around vec and SVD.
        """
        if k <= 1:
            u = vec.reshape(-1, 1)
            return normalize(u, axis=0)

        d = vec.shape[0]
        rng = np.random.RandomState(self.random_state)
        # stack the vector and k-1 noise perturbed copies
        M = np.zeros((k, d), dtype=float)
        M[0, :] = vec
        for i in range(1, k):
            noise = rng.normal(loc=0.0, scale=1e-3, size=d)
            M[i, :] = vec + noise
        # orthonormalize columns (we need basis in embedding space)
        # SVD on M.T gives V (d x r) basis in embedding space
        U_m, S_m, Vt_m = np.linalg.svd(M, full_matrices=False)
        # Vt_m: (r x d) -> Vt_m.T: (d x r)
        V = Vt_m.T
        basis = V[:, :k]
        # ensure normalized columns
        basis = normalize(basis, axis=0)
        return basis

    # -------------------------
    # Build angle + spectral features between prompt basis and answer basis
    # -------------------------
    def _angle_and_spectral_features(self, Ua: np.ndarray, Ub: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Returns: sigma array and combined feature dict (spectral + angle)
        """
        s = principal_singulars(Ua, Ub)
        spec = spectral_features_from_sigma(s)
        ang = angle_basic_from_sigma(s)
        # merge preserving order (spectral keys then angle keys)
        merged = {}
        merged.update(spec)
        merged.update(ang)
        return s, merged

    # -------------------------
    # Fit: build TF-IDF, global SVD, prompt subspace, train ridge
    # -------------------------
    def fit(self, question: List[str], answers: List[str], scores: List[float]):
        """
        answers: list of answer strings (training set)
        scores: numeric list (one score per answer)
        """
        print("Fitting prompt regressor")

        # TF-IDF fit on prompt + answers
        self.tfidf_ = TfidfVectorizer(lowercase=True, stop_words=[self.stop_words])
        corpus = [self.prompt] + answers + question
        self.tfidf_.fit(corpus)

        self.garbage_detector = GarbageDetector(self.tfidf_)

        # Global SVD embedding space fit
        X_tfidf = self.tfidf_.transform(corpus)  # sparse matrix
        self.svd_ = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        self.svd_.fit(X_tfidf)

        # Prompt embedding and prompt subspace (Ua)
        P_tfidf = self.tfidf_.transform([self.prompt])
        P_emb = self.svd_.transform(P_tfidf)[0]
        self.U_prompt_ = self._make_local_subspace(P_emb, k=max(1, self.answer_subspace_dim))

        # Build feature matrix for all training answers
        feature_rows = []
        for question, ans in zip(question, answers):
            tf = self.tfidf_.transform([ans])
            a_emb = self.svd_.transform(tf)[0]
            q_emb = self.svd_.transform(self.tfidf_.transform([question]))[0]

            U_ans = self._make_local_subspace(a_emb, k=self.answer_subspace_dim)
            U_question = self._make_local_subspace(q_emb, k=self.answer_subspace_dim)

            sigma, feat_dict = self._angle_and_spectral_features(U_question, U_ans)
            # record angle feature names on first iteration for ordering
            if not self.angle_feature_names_:
                self.angle_feature_names_ = list(feat_dict.keys())
            # arrange feature vector: [embedding coords..., angle_feats in recorded order]
            angle_vector = np.array([feat_dict[k] for k in self.angle_feature_names_], dtype=float)
            row = np.concatenate([a_emb, angle_vector])
            feature_rows.append(row)

        X = np.vstack(feature_rows) if feature_rows else np.zeros(
            (0, self.n_components + len(self.angle_feature_names_)))
        y = np.asarray(scores, dtype=float)

        # Train ridge regressor (more stable than plain linear)
        self.model_ = Ridge(alpha=self.model_alpha)
        if X.shape[0] > 0:
            self.model_.fit(X, y)
        return self

    # -------------------------
    # transform: produce feature matrix for answers
    # -------------------------
    def transform(self, answers: List[str]) -> np.ndarray:
        """
        Returns matrix of shape (n_answers, n_components + n_angle_features)
        """
        check_is_fitted(self, ["tfidf_", "svd_", "U_prompt_", "model_"])
        rows = []
        for ans in answers:
            tf = self.tfidf_.transform([ans])
            emb = self.svd_.transform(tf)[0]
            U_ans = self._make_local_subspace(emb, k=self.answer_subspace_dim)
            sigma, feat_dict = self._angle_and_spectral_features(self.U_prompt_, U_ans)
            angle_vector = np.array([feat_dict[k] for k in self.angle_feature_names_], dtype=float)
            row = np.concatenate([emb, angle_vector])
            rows.append(row)
        return np.vstack(rows)

    # -------------------------
    # predict
    # -------------------------
    def predict(self, answers: List[str]) -> np.ndarray:
        check_is_fitted(self, ["model_"])

        predictions = []
        for i in range(len(answers)):
            print(f"{self.name}: Proccessing answer {i} of {len(answers)}")
            answer = answers[i]
            if self.garbage_detector.is_garbage(answer):
                predictions.append(self.garbage_penalty)
                continue

            X = self.transform([answer])
            predict = self.model_.predict(X)
            predictions.append(predict[0])

        return np.array(predictions)

    # -------------------------
    # explain: fully detailed metrics for a single answer
    # -------------------------
    def explain(self, answer: str) -> Dict[str, Any]:
        """
        Returns:
          - embedding: vector
          - sigma: array of singular values
          - angles_rad: arccos(sigma)
          - angle_features: dict (spectral + primary-angle features)
          - model_input_vector: concatenated feature vector
          - prediction: float
        """
        check_is_fitted(self, ["tfidf_", "svd_", "U_prompt_", "model_"])
        tf = self.tfidf_.transform([answer])
        emb = self.svd_.transform(tf)[0]
        U_ans = self._make_local_subspace(emb, k=self.answer_subspace_dim)
        s = principal_singulars(self.U_prompt_, U_ans)
        angle_feats = spectral_features_from_sigma(s)
        angle_basic = angle_basic_from_sigma(s)
        angle_feats.update(angle_basic)
        angles = np.arccos(s) if s.size > 0 else np.array([])

        # model input vector (embed + ordered angle features)
        angle_vector = np.array([angle_feats[k] for k in self.angle_feature_names_],
                                dtype=float) if self.angle_feature_names_ else np.array([])
        model_input = np.concatenate([emb, angle_vector]) if angle_vector.size > 0 else emb
        pred = float(self.model_.predict([model_input])[0])
        return {
            "embedding": emb,
            "sigma": s,
            "angles_rad": angles,
            "angle_features": angle_feats,
            "model_input_vector": model_input,
            "prediction": pred
        }

    # -------------------------
    # utility
    # -------------------------
    def get_feature_names(self) -> List[str]:
        """Return names for columns in transform(X) output."""
        check_is_fitted(self, ["tfidf_", "svd_", "U_prompt_", "model_"])
        emb_names = [f"svd_{i}" for i in range(self.n_components)]
        angle_names = list(self.angle_feature_names_)
        return emb_names + angle_names

    @staticmethod
    def get_and_fit(
            prompt: str,
            answers: List[str],
            questions: List[str],
            scores: List[float],
            n_components: int = 100,
            answer_subspace_dim: int = 1,
            model_alpha: float = 1.0,
            stop_words: Optional[str] = "russian",
            random_state: int = 42,
            garbage_penalty: float = 0.0,
    ):
        model = PromptAngleRegressorAdvanced(
            prompt=prompt,
            n_components=n_components,
            answer_subspace_dim=answer_subspace_dim,
            model_alpha=model_alpha,
            stop_words=stop_words,
            random_state=random_state,
            garbage_penalty=garbage_penalty,
        )
        model.fit(questions, answers, scores)

        return model
