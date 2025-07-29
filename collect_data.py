from utils import scraping, build_dataset


def collect_data(path: str = "scraped_articles"):
    #scraping.scrape(output_dir=path)
    build_dataset.build_dataset(data_dir=path, output_file="dataset/dataset3.npy")

if __name__ == "__main__":
    collect_data("learning_data")
    print("Data collection completed.")