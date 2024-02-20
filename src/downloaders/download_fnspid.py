from datasets import load_dataset, DownloadConfig

dataset = load_dataset("Zihan1004/FNSPID", download_config=DownloadConfig(resume_download=True))
print(dataset)

print(dataset["train"][:1])
print(dataset[:1])
# dataset.save_to_disk("C:/Users/Adria/Documents/Github Projects/data/news_trading/fnspid.hf")
