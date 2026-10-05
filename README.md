GuessArt - Bachelor's Thesis in Computer Science

Bachelor's thesis project for artistic style classification using Deep Learning.

The project uses three computer vision architectures, ResNet50V2, DenseNet121, and Vision Transformer (ViT), trained on the open-source WikiArt dataset, with a selection of 10 artistic styles.

Reproducibility note: the original notebook was developed in a Kaggle environment and contains /kaggle/input and /kaggle/working paths. For this reason, the notebook can be viewed directly on GitHub, but fully reproducing the training requires recreating the dataset/environment and updating the local or Kaggle paths.

Project Contents

The notebook contains the main stages of the project:

1. dataset preparation and balancing;
2. selection of 10 artistic styles;
3. resizing of images to 224×224;
4. splitting into training, validation, and test sets;
5. construction and training of ResNet50V2;
6. fine-tuning of ResNet50V2;
7. construction and training of DenseNet121;
8. fine-tuning of DenseNet121;
9. construction and training of Vision Transformer (ViT);
10. fine-tuning of ViT;
11. evaluation using metrics and confusion matrix;
12. Grad-CAM visualization for ResNet50V2 and DenseNet121;
13. attention map visualization for ViT;
14. web application developed with Streamlit.

Dataset

The system was trained on the open-source WikiArt dataset available in the Kaggle environment used for the project.

The 10 target styles are:

* Early Renaissance
* Baroque
* Romanticism
* Impressionism
* Symbolism
* Expressionism
* Cubism
* Ukiyo-e
* Abstract Expressionism
* Pop Art

The notebook uses 1,100 images per style, for a theoretical total of 11,000 images.

The planned split is 80% for training, 10% for validation, and 10% for testing.

 Models

-ResNet50V2

The notebook uses ResNet50V2 with ImageNet weights, initially frozen and subsequently fine-tuned on the final layers.

-DenseNet121

DenseNet121 is used with a similar procedure: initial feature extraction followed by fine-tuning.

-Vision Transformer

The project uses vit-keras with a ViT-B/16, followed by a feature extraction phase and fine-tuning.

Metrics and Explainability

The notebook uses, among others, accuracy and top-2 accuracy and produces analysis tools such as confusion matrices and classification reports.

For interpretability, the following techniques are used:

* Grad-CAM for ResNet50V2 and DenseNet121;
* Attention Map for the Vision Transformer.

Streamlit Application

app.py contains the web application developed as part of the project. The application allows users to upload an image and obtain predictions from the three models, together with their respective confidence information and explainability visualizations.

 Important Before Running

The original application uses the following Kaggle paths for the models:

/kaggle/input/models/emanueladagostino/resnet50/keras/default/1
/kaggle/input/models/emanueladagostino/dense121/keras/default/1
/kaggle/input/models/emanueladagostino/vitkeras/keras/default/1


If the application is run outside Kaggle, these paths must be modified according to the location of the models in the selected environment.

Running the Application

After installing the dependencies:

bash
streamlit run app.py


Installing Dependencies

An initial list of dependencies is available in requirements.txt. The versions have not been artificially reconstructed from the notebook: the original notebook does not contain a fully versioned environment.

For an execution faithful to the thesis project, it is recommended to use an environment compatible with TensorFlow/Keras and with the version of vit-keras used during development.

Weights & Biases

The notebook uses Weights & Biases for training monitoring.

Never include API keys, tokens, or other secrets in the repository. The notebook uses Kaggle Secrets to access WANDB_API_KEY; the key must not be copied into public code.

Kaggle and Notebook

The guessart-tesi.ipynb file is the original project notebook. To run it again in Kaggle, the dataset and any required models must be made available at the expected locations, or the corresponding paths must be modified.

Large Files and Artifacts

To avoid making the repository unnecessarily large, the following are not included:

* WikiArt dataset;
* .keras model files;
* model.weights.h5 and config.json files for models saved externally;
* outputs generated during training;
* credentials or tokens.

These elements can be added separately through appropriate storage or artifact/model tracking systems, if necessary.
