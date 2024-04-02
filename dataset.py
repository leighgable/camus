import datasets


data_path = "dataset"
DATASET = f"{data_path}/hugging_doc"

def load_dataset(db_name="m-ric/huggingface_doc", db_path=data_path):
    # Check if the directory already exists
    if not os.path.exists(db_path):
        # Create the directory
        os.makedirs(db_path)
    ds = datasets.load_dataset(db_name, split="train")
    ds.save_to_disk(DATASET)

load_dataset()
