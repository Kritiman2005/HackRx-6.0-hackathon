from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    token="your_HF_token_here"
)
