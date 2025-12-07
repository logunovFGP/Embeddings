import json
import os

from dotenv import load_dotenv, find_dotenv
from langchain_gigachat import GigaChat, GigaChatEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

from src.dialogue.history import generate_from_llm, read_from_file
from src.llm.agent import Agent
from src.llm.prompts import ValidatorPrompt, AnalyticsPrompt, \
    FinancePrompt, BadGuyPrompt
from src.predict_models.embedding_regressor import EmbeddingRegressor
from src.predict_models.prompt_regressor_v2 import PromptAngleRegressorAdvanced
from src.utils import embedding_metrics, save_to_file, NumpyArrayEncoder

# загружаем переменные окружения из .env
load_dotenv(find_dotenv())

PATH_TO_DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Settings for history of llm chat
SIZE_FOR_GENERATE = 100
TEST_HISTORY_FROM_FILE = True
TRAIN_HISTORY_FROM_FILE = True

TRAIN_FILE = "train_history_100.json"
TEST_FILE = "test_history_100_with_bad_guy(2).json"

# Settings for TF-IDT
NGRAM_RANGE = (1, 2)
STOP_WORDS = ["russian"]
MAX_FEATURES = 5000
LOWERCASE = True

# Settings for SVD
N_COMPONENTS = 100
RANDOM_STATE = 42


def make_history(score_fn, bad_guy: Agent):
    start_message = """Если смотреть на кризис 2008 года системно, то ключевым триггером стала переоценка рисков на рынке ипотечных деривативов. 
    Банки массово упаковывали плохие кредиты в сложные финансовые инструменты, теряя понимание их реальной стоимости…"""

    path_to_train = os.path.join(PATH_TO_DATA, TRAIN_FILE)
    if not TRAIN_HISTORY_FROM_FILE:
        train_history = generate_from_llm(analytic, finance, bad_guy, start_message, SIZE_FOR_GENERATE, score_fn)
        train_history.save_to_file(path_to_train)
    else:
        train_history = read_from_file(path_to_train)

    path_to_test = os.path.join(PATH_TO_DATA, TEST_FILE)
    if not TEST_HISTORY_FROM_FILE:
        test_history = generate_from_llm(analytic, finance, bad_guy, start_message, SIZE_FOR_GENERATE, score_fn)
        test_history.save_to_file(path_to_test)
    else:
        test_history = read_from_file(path_to_test)

    return train_history, test_history


def get_predict_models_for_agent(
        prompt: str,
        model: Agent,
        embeddings,
        questions,
        answers,
        train_score,
        test_questions
) -> list:
    return [
        PromptAngleRegressorAdvanced.get_and_fit(
            prompt=prompt,
            answers=answers,
            questions=questions,
            scores=train_score,
            answer_subspace_dim=10
        ),
        # LLMRegressor.get_and_fit(
        #     model=model,
        #     prompt=prompt,
        # ),
        EmbeddingRegressor.get_and_fit(
            embedding=lambda docs: embeddings.embed_documents(docs),
            questions=test_questions,
            prompt=prompt,
        )
    ]


if __name__ == "__main__":
    model = GigaChat(
        credentials=os.getenv("GIGACHAT_API_TOKEN"),
        model=os.getenv("GIGACHAT_MODEL"),
        scope=os.getenv("GIGACHAT_SCOPE"),
        verify_ssl_certs=False,
    )

    model_ollama = ChatOllama(model="deepseek-r1:1.5b")
    embedding_ollama = OllamaEmbeddings(model="nomic-embed-text:latest")

    embeddings = GigaChatEmbeddings(
        credentials=os.getenv("GIGACHAT_API_TOKEN"),
        scope=os.getenv("GIGACHAT_SCOPE"),
        verify_ssl_certs=False
    )

    validator = Agent(
        name=ValidatorPrompt.name,
        prompt=ValidatorPrompt.system,
        model=model,
    )
    analytic = Agent(
        name=AnalyticsPrompt.name,
        prompt=AnalyticsPrompt.system,
        model=model
    )
    finance = Agent(
        name=FinancePrompt.name,
        prompt=FinancePrompt.system,
        model=model
    )
    bad_guy = Agent(
        name=BadGuyPrompt.name,
        prompt=BadGuyPrompt.system,
        model=model
    )

    dialogues = [
        (FinancePrompt, AnalyticsPrompt),
        (AnalyticsPrompt, FinancePrompt),
    ]

    result = {}
    train_history, test_history = make_history(
        lambda prompt, response: embedding_metrics(prompt, response, embeddings),
        bad_guy
    )
    for question, answers in dialogues:
        predict_models = get_predict_models_for_agent(
            prompt=answers.system,
            model=validator,
            embeddings=embedding_ollama,
            questions=train_history.messages_by_author(question.name),
            answers=train_history.messages_by_author(answers.name),
            train_score=train_history.normalize_scores_by_author(answers.name),
            test_questions=test_history.messages_by_author(question.name)
        )

        actor_answers = test_history.messages_by_author(answers.name)
        result[answers.name] = {
            model.name: {"Predict scores": list(zip(model.predict(actor_answers), actor_answers))}
            for model in predict_models
        }

    json_result = json.dumps(result, indent=2, cls=NumpyArrayEncoder, ensure_ascii=False)
    save_to_file(lambda file: file.write(json_result), f"../results.json")
