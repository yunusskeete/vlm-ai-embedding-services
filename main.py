def run_tests() -> None: ...


if __name__ == "__main__":
    import os
    from pathlib import Path

    import uvicorn
    from dotenv import load_dotenv

    assert load_dotenv(dotenv_path=Path(".env"))

    uvicorn.run(
        "app:app",
        host=os.environ.get("INFERENCE_HOST", "localhost"),
        port=int(os.environ.get("INFERENCE_PORT", 8004)),
        reload=(os.environ.get("RELOAD_APP", "true").lower() == "true"),
    )
