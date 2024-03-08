# Auto-categorization Pipeline for Vistaar Trade
As Nepal's pioneering B2B platform, Vistaar Trade aimed to streamline its product categorization process, encompassing an extensive catalog of over 6000 subcategories. My role as an AI Solutions Engineer was pivotal in transitioning from a manual classification system to an automated, efficient model. I was able to achieve an average f1-score across all models of 0.62, which is very promising when taking into account how many categories there are.

## Leveraging NLP for Enhanced Word Embeddings
At the core of the auto-categorization pipeline is a sophisticated NLP framework capable of vectorizing product names for AI processing. After experimenting with various models, I chose FastText for its ability to handle Out-of-Vocabulary (OOV) words, a common challenge in dynamic product inventories.

Through meticulous pre-processing — including lemmatization, stopword removal, special character filtering, and tokenization — I converted product names into token lists. The aggregate of FastText-generated word embeddings for these tokens served as the foundation for product classification.

## Hierarchical Neural Network Structure
To manage the sheer scale of potential subcategories, I architected a hierarchical neural network system. Initially, a comprehensive BERT model was considered; however, computational demands led to a more refined approach:

Level 1: A primary neural network, trained on a vast dataset, classifies products into one of 50 overarching categories.
Level 2: Depending on the outcome of Level 1, one of 15 specialized neural networks takes over, assigning the product to one of the 6000+ subcategories, based on the established word embeddings.

<div align="center">
  <img src="/algorithm.png" alt="Pipeline"><br>
  <em>Fig: Pipeline</em><br><br>
</div>

## Example
To illustrate the efficacy of this solution, I've provided access to a model trained specifically for the "Building and Construction" category, which includes 652 subcategories. <br>
**Setup**: Clone the repository. Then, create and activate a new environment, install necessary dependencies listen on requirements.txt. <br>
**Data Preparation**: Compile your product data into a DataFrame with a single column titled 'Product Name'. Save this as a CSV in the base directory. <br>
**Execution**: Navigate to the base directory and run the start.py script with your CSV file:

cd /path/to/base/directory # Ensure you're in the base directory <br>
python3 src/start.py your_product_file.csv

## Looking Ahead
Currently, my focus is on deploying all 16 models onto AWS via RestAPI, optimizing for scalability and ease of use. For confidentiality reasons, only a fraction of the models are showcased here. This serves as a testament to the robustness of the auto-categorization pipeline.


