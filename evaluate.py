"""
evaluate.py
Evaluates the RAG QA system against a small test set of questions
with known/expected answers, using an LLM as a judge for correctness.
"""

import os
import json
from generation.llm_chain import build_qa_chain, ask_question
from retrieval.vectorstore import load_vectorstore
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Config ---
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
JUDGE_MODEL = "gpt-4o-mini"


def load_test_set(path: str = TEST_SET_PATH):
    """
    Load evaluation questions from a JSON file.
    Expected format:
    [
      {"question": "...", "expected_answer": "..."},
      ...
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def judge_answer(question: str, expected: str, actual: str) -> dict:
    """
    Use an LLM to judge whether the actual answer matches the expected answer
    in meaning (not exact wording).
    """
    judge = ChatOpenAI(model=JUDGE_MODEL, temperature=0)

    prompt = f"""
You are grading a QA system's answer.

Question: {question}
Expected answer: {expected}
System's answer: {actual}

Does the system's answer correctly convey the same information as the expected answer?
Reply with ONLY a JSON object in this exact format:
{{"correct": true or false, "reasoning": "one short sentence"}}
"""
    response = judge.invoke(prompt)
    content = response.content.strip()

    # Strip markdown code fences if the model adds them
    content = content.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"correct": False, "reasoning": f"Could not parse judge output: {content}"}


def run_evaluation():
    """Run the full evaluation loop and print a summary report."""
    print("Loading vector store and QA chain...")
    vectorstore = load_vectorstore()
    qa_chain = build_qa_chain(vectorstore)

    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test question(s)\n")

    results = []
    correct_count = 0

    for i, item in enumerate(test_set, start=1):
        question = item["question"]
        expected = item["expected_answer"]

        answer, sources = ask_question(qa_chain, question)
        verdict = judge_answer(question, expected, answer)

        is_correct = verdict.get("correct", False)
        correct_count += int(is_correct)

        results.append({
            "question": question,
            "expected_answer": expected,
            "system_answer": answer,
            "sources": sources,
            "correct": is_correct,
            "reasoning": verdict.get("reasoning", ""),
        })

        status = "PASS" if is_correct else "FAIL"
        print(f"[{i}/{len(test_set)}] {status} — {question}")

    accuracy = correct_count / len(test_set) if test_set else 0
    print(f"\n=== Evaluation Summary ===")
    print(f"Accuracy: {correct_count}/{len(test_set)} ({accuracy:.1%})")

    # Save detailed results
    output_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to {output_path}")

    return results


if __name__ == "__main__":
    run_evaluation()


    [
  {
    "question": "What is the main topic of the document?",
    "expected_answer": "A short factual answer based on your actual PDF content."
  },
  {
    "question": "Another question about your document",
    "expected_answer": "Its expected answer"
  }
]