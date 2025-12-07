from collections.abc import Callable

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingRegressor:
    name: str = "EmbeddingRegressor"

    def __init__(
            self,
            embedding: Callable[[list[str]], list],
            chunk_size=500,
            chunk_overlap=14
    ):
        self.embedding = embedding
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.embs = np.array([])

    def _get_embedding(self, content):
        split = self.splitter.split_text(content)
        embedding = self.embedding(split)
        return np.array(embedding)

    def fit(self, questions, prompt):
        print("Fitting embedding regressor")

        emb_prompt = self._get_embedding(prompt)
        self.embs = [self._get_embedding(question) for question in questions]
        return self

    def predict(self, answers: list) -> np.ndarray:
        result = []
        for i in range(len(answers)):
            print(f"{self.name}: Proccessing answer {i} of {len(answers)}")
            answer = answers[i]

            emb_question = self.embs[i]
            emb_answer = self._get_embedding(answer)

            cos_sim = cosine_similarity(emb_question, emb_answer).mean(axis=0)[0]
            result.append(cos_sim)

        return np.array(result)

    @staticmethod
    def get_and_fit(embedding: Callable[[list[str]], list], questions: list, prompt: str):
        model = EmbeddingRegressor(embedding=embedding)
        model.fit(questions, prompt)
        return model
