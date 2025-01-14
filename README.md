# ApolloDFT  

ApolloDFT (Deep Fake Transformers) is a suite of advanced deep learning models designed to detect deepfakes. Currently, we are releasing an AI-generated text detector, with support for other media types (e.g., images) planned for future updates.  

## AI Text Detector  
The Apollo AI Text Detector combines zero-shot perplexity-based techniques with supervised methods, delivering superior performance and generalization capabilities. For detailed information on the Apollo detector and its performance, refer to [apollo-text.md](apollo-text.md).  

We are also open-sourcing the supervised component of Apollo, a fine-tuned RoBERTa classifier, available for download from [Hugging Face](https://huggingface.co/fakespotailabs/roberta-base-ai-text-detection-v1) under the Apache-2.0 license.  

## Firefox Extension  
The Apollo Text Detector is also accessible through our [Firefox Extension](https://addons.mozilla.org/en-US/firefox/addon/deep-fake-detector/). Simply highlight any text while browsing the internet to receive a confidence score estimating the likelihood that the text was generated by AI. For additional information, visit the [Welcome Page](https://www.fakespot.com/dfd).  

## Version  
We are continually enhancing the detector. For detailed information about updates, refer to [version.md](version.md). Please note that the Firefox extension always utilizes the latest version of the model.  

## Comments and Suggestions  
Like any machine learning model, the Apollo detector makes errors, especially with the challenging task of deepfake detection. We are constantly working to enhance its accuracy, and your feedback is invaluable in this process. Feel free to share your comments or questions via the Issue page of this repository or by [email](mailto:contact@fakespot.com). You can also provide feedback through the extension itself or leave a review on our [extension page](https://addons.mozilla.org/en-US/firefox/addon/deep-fake-detector/).  
