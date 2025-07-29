import trafilatura
import os
from datetime import datetime

# Your list of URLs (make sure these are article/blog/news pages that allow scraping)
URLS = [
    "https://en.wikipedia.org/wiki/Large_language_model",
    "https://en.wikipedia.org/wiki/Human",
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Neural_network",
    "https://en.wikipedia.org/wiki/Natural_language_processing",
    "https://en.wikipedia.org/wiki/Deep_learning",
    "https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)",
    "https://en.wikipedia.org/wiki/Data_science",
    "https://en.wikipedia.org/wiki/Computer_programming",
    "https://en.wikipedia.org/wiki/Software_engineering",
    "https://en.wikipedia.org/wiki/Python_(programming_language)",
    "https://en.wikipedia.org/wiki/History_of_artificial_intelligence",
    "https://en.wikipedia.org/wiki/ChatGPT",
    "https://en.wikipedia.org/wiki/OpenAI",
    "https://en.wikipedia.org/wiki/Algorithm",
    "https://en.wikipedia.org/wiki/Computer_science",
    "https://en.wikipedia.org/wiki/Big_data",
    "https://en.wikipedia.org/wiki/Computer_vision",
    "https://en.wikipedia.org/wiki/Robotics",
    "https://en.wikipedia.org/wiki/Internet_of_things",
    "https://en.wikipedia.org/wiki/Data_mining",
    "https://en.wikipedia.org/wiki/Quantum_computing",
    "https://en.wikipedia.org/wiki/Cloud_computing",
    "https://en.wikipedia.org/wiki/Computer_security",
    "https://en.wikipedia.org/wiki/Software_testing",
    "https://en.wikipedia.org/wiki/Programming_language",
    "https://en.wikipedia.org/wiki/Compiler",
    "https://en.wikipedia.org/wiki/Operating_system",
    "https://en.wikipedia.org/wiki/Distributed_computing",
    "https://en.wikipedia.org/wiki/Virtual_reality",
    "https://en.wikipedia.org/wiki/Augmented_reality",
    "https://en.wikipedia.org/wiki/Blockchain",
    "https://en.wikipedia.org/wiki/Cryptocurrency",
    "https://en.wikipedia.org/wiki/Data_structure",
    "https://en.wikipedia.org/wiki/Database",
    "https://en.wikipedia.org/wiki/Artificial_neural_network",
    "https://en.wikipedia.org/wiki/Reinforcement_learning",
    "https://en.wikipedia.org/wiki/Supervised_learning",
    "https://en.wikipedia.org/wiki/Unsupervised_learning",
    "https://en.wikipedia.org/wiki/Semantic_web",
    "https://en.wikipedia.org/wiki/Computer_graphics",
    "https://en.wikipedia.org/wiki/Human–computer_interaction",
    "https://en.wikipedia.org/wiki/Information_retrieval",
    "https://en.wikipedia.org/wiki/Information_theory",
    "https://en.wikipedia.org/wiki/Knowledge_representation_and_reasoning",
    "https://en.wikipedia.org/wiki/Logic_programming",
    "https://en.wikipedia.org/wiki/Machine_translation",
    "https://en.wikipedia.org/wiki/Software_architecture",
    "https://en.wikipedia.org/wiki/Software_development_process",
    "https://en.wikipedia.org/wiki/Computer_network",
    "https://en.wikipedia.org/wiki/Parallel_computing",
    "https://en.wikipedia.org/wiki/Computer_simulation",
    "https://en.wikipedia.org/wiki/Data_analytics",
    "https://en.wikipedia.org/wiki/Data_visualization",
    "https://en.wikipedia.org/wiki/Evolutionary_computation",
    "https://en.wikipedia.org/wiki/Expert_system",
    "https://en.wikipedia.org/wiki/Fuzzy_logic",
    "https://en.wikipedia.org/wiki/Genetic_algorithm",
    "https://en.wikipedia.org/wiki/Graph_theory",
    "https://en.wikipedia.org/wiki/Heuristic",
    "https://en.wikipedia.org/wiki/Internet",
    "https://en.wikipedia.org/wiki/Logic",
    "https://en.wikipedia.org/wiki/Mathematics",
    "https://en.wikipedia.org/wiki/Neuroscience",
    "https://en.wikipedia.org/wiki/Pattern_recognition",
    "https://en.wikipedia.org/wiki/Probability_theory",
    "https://en.wikipedia.org/wiki/Programming_paradigm",
    "https://en.wikipedia.org/wiki/Recursion",
    "https://en.wikipedia.org/wiki/Software_engineer",
    "https://en.wikipedia.org/wiki/Statistics",
    "https://en.wikipedia.org/wiki/System_analysis",
    "https://en.wikipedia.org/wiki/System_administration",
    "https://en.wikipedia.org/wiki/Theoretical_computer_science",
    "https://en.wikipedia.org/wiki/Unix",
    "https://en.wikipedia.org/wiki/Virtual_machine",
    "https://en.wikipedia.org/wiki/Web_development",
    "https://en.wikipedia.org/wiki/Web_scraping",
    "https://en.wikipedia.org/wiki/XML",
    "https://en.wikipedia.org/wiki/JSON",
    "https://en.wikipedia.org/wiki/HTML",
    "https://en.wikipedia.org/wiki/CSS",
    "https://en.wikipedia.org/wiki/JavaScript",
    "https://en.wikipedia.org/wiki/TypeScript",
    "https://en.wikipedia.org/wiki/Node.js",
    "https://en.wikipedia.org/wiki/React_(JavaScript_library)",
    "https://en.wikipedia.org/wiki/Angular_(web_framework)",
    "https://en.wikipedia.org/wiki/Vue.js",
    "https://en.wikipedia.org/wiki/Bootstrap_(front-end_framework)",
    "https://en.wikipedia.org/wiki/Django_(web_framework)",
    "https://en.wikipedia.org/wiki/Flask_(web_framework)",
    "https://en.wikipedia.org/wiki/Ruby_on_Rails",
    "https://en.wikipedia.org/wiki/Software_license",
    "https://en.wikipedia.org/wiki/Open_source",
    "https://en.wikipedia.org/wiki/Free_software",
    "https://en.wikipedia.org/wiki/Linux",
    "https://en.wikipedia.org/wiki/Git",
    "https://en.wikipedia.org/wiki/Version_control",
    "https://en.wikipedia.org/wiki/Agile_software_development",
    "https://en.wikipedia.org/wiki/Scrum_(software_development)",
    "https://en.wikipedia.org/wiki/DevOps",
]

LoveUrl = [
    "https://en.wikipedia.org/wiki/Love",
    "https://en.wikipedia.org/wiki/Romantic_love",
    "https://en.wikipedia.org/wiki/Human",
    "https://en.wikipedia.org/wiki/Sexual_love",
    "https://en.wikipedia.org/wiki/Emotion",
]

def scrape(output_dir: str = "scraped_articles"):
    # Create output directory
    OUTPUT_DIR = output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Loop through URLs
    for i, url in enumerate(LoveUrl, start=1):
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            if text:
                filename = f"article_{i}.txt"
                filepath = os.path.join(OUTPUT_DIR, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"✅ Saved: {filename}")
            else:
                print(f"⚠️ Could not extract text from {url}")
        else:
            print(f"❌ Failed to download {url}")
