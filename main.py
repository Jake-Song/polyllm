import argparse
import sys

from polyllm import PolyLLM


def main():
    parser = argparse.ArgumentParser(
        description="Call OpenAI, Claude, or Gemini through a single interface."
    )
    parser.add_argument(
        "provider", choices=["openai", "claude", "gemini"], help="LLM provider to call"
    )
    parser.add_argument("prompt", help="The prompt to send")
    parser.add_argument("--model", default=None, help="Override the provider's default model")
    parser.add_argument("--system", default=None, help="Optional system prompt")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--effort",
        default=None,
        help="Thinking effort, e.g. low/medium/high (levels vary by provider)",
    )
    parser.add_argument(
        "--raw", action="store_true", help="Print the raw API response as JSON"
    )
    args = parser.parse_args()

    llm = PolyLLM(provider=args.provider, model=args.model)
    try:
        result = llm.chat(
            args.prompt,
            system=args.system,
            max_tokens=args.max_tokens,
            effort=args.effort,
            raw=args.raw,
        )
    except Exception as e:
        print(f"Error calling {args.provider}: {e}", file=sys.stderr)
        sys.exit(1)

    print(result)


if __name__ == "__main__":
    main()
