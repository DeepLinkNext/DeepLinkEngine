"""Call the DeepLinkEngine chat API using only the Python standard library."""
import argparse
import json
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="你好，请介绍一下自己")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="deeplink")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps({"model": args.model, "messages": [{"role": "user", "content": args.prompt}],
                         "max_tokens": args.max_tokens, "temperature": 0,
                         "chat_template_kwargs": {"enable_thinking": False}}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)
    print(result["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
