"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --chat     # 会話履歴を保ったまま続けて質問する（宿題・実装）

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=question)
    print_usage(response)
    return response.text


def print_usage(response) -> None:
    """response.usage_metadata から入力・出力トークン数を表示する（第1回 演習）。"""
    usage = response.usage_metadata
    if usage is None:
        return
    print(f"[tokens] 入力 {usage.prompt_token_count} / 出力 {usage.candidates_token_count}")


def chat_loop(first_question: str | None = None) -> None:
    """会話履歴を保ったまま続けて質問する（第1回 宿題・実装）。

    client.chats が履歴を保持し、毎回それまでのやり取りを含めてモデルに送る。
    空行・exit・quit で終了する。
    """
    client = genai.Client(api_key=get_api_key())
    chat = client.chats.create(model=MODEL)
    question = first_question
    while True:
        if question is None:
            try:
                question = input("\n[質問] > ").strip()
            except EOFError:
                break
        if question in ("", "exit", "quit"):
            break
        try:
            response = chat.send_message(question)
        except errors.APIError as e:
            print(api_error_message(e.code, e.message))
        else:
            print_usage(response)
            print(response.text)
            print(f"[history] {len(chat.get_history())} 件")
        question = None


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=None)
    parser.add_argument("--chat", action="store_true", help="会話履歴を保ったまま続けて質問する")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    if args.chat:
        chat_loop(args.question)
        return
    args.question = args.question or DEFAULT_QUESTION
    print(f"[質問] {args.question}\n")
    try:
        print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
