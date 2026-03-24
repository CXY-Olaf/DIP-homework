import os


def _append_no_proxy(hosts):
    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        items = [item.strip() for item in current.split(",") if item.strip()]
        for host in hosts:
            if host not in items:
                items.append(host)
        os.environ[key] = ",".join(items)


def launch_demo(demo, port=None):
    # Some Windows/proxy setups incorrectly route localhost checks through a proxy.
    _append_no_proxy(["127.0.0.1", "localhost", "0.0.0.0"])

    launch_kwargs = {
        "inbrowser": True,
        "show_error": True,
    }
    if port is not None:
        launch_kwargs["server_port"] = port

    try:
        return demo.launch(server_name="127.0.0.1", **launch_kwargs)
    except ValueError as exc:
        if "localhost is not accessible" not in str(exc):
            raise
        print("Localhost check failed. Retrying with a Gradio share link...")
        return demo.launch(share=True, **launch_kwargs)
