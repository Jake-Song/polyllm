import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Serve the polyllm web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Reload on code changes")
    args = parser.parse_args()

    print(f"polyllm web UI on http://{args.host}:{args.port}")
    uvicorn.run(
        "polyllm.web.server:app", host=args.host, port=args.port, reload=args.reload
    )


if __name__ == "__main__":
    main()
