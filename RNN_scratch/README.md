This repository implements a simple Recurrent Neural Network (RNN) with many-to-many architecture. \n
The architecture and equations used are based on GoodFellow, Yoshua Bengio, and Aaron Courville's 2016 book titled [Deep Learning](https://www.deeplearningbook.org/)
### Recurrent Neural Network architecture
![Recurrent Neural Network architecture Page - 370](https://github.com/choudharynishu/ml_research_papers/blob/d5d054ccc8336dbb8a80cba3d0aa57537e8eee99/RNN_scratch/definition/definition_RNN%20Small.png)
### Parameter definitions
![Parameters](https://github.com/choudharynishu/ml_research_papers/blob/d5d054ccc8336dbb8a80cba3d0aa57537e8eee99/RNN_scratch/definition/definition_parameters%20Small.png)
### Prediction 
![Prediction definition](https://github.com/choudharynishu/ml_research_papers/blob/d5d054ccc8336dbb8a80cba3d0aa57537e8eee99/RNN_scratch/definition/definition_prob_dict%20Small.png)
### Loss function (Cross-entropy loss)
![Loss function defintion](https://github.com/choudharynishu/ml_research_papers/blob/main/RNN_scratch/definition/definition_loss_value%20Small.png)

### Backpropagation Through Time (BPTT) - derivatives
![Derivative of Loss function w.r.t Prediction](https://github.com/choudharynishu/ml_research_papers/blob/main/RNN_scratch/definition/derivative_d_prob%20Small.png)

![Derivative of Loss-function w.r.t Hidden State](https://github.com/choudharynishu/ml_research_papers/blob/main/RNN_scratch/definition/derivative_dh%20Small.png)
![Derivative of Loss-function w.r.t V,c, and b](https://github.com/choudharynishu/ml_research_papers/blob/main/RNN_scratch/definition/derivative_dc_db_dV%20Small.png)
![Derivative of Loss-function w.r.t W and U](https://github.com/choudharynishu/ml_research_papers/blob/main/RNN_scratch/definition/derivative_dU_dW%20Small.png)
Reference: 
```
@book{Goodfellow-et-al-2016,
    title={Deep Learning},
    author={Ian Goodfellow and Yoshua Bengio and Aaron Courville},
    publisher={MIT Press},
    note={\url{http://www.deeplearningbook.org}},
    year={2016}
}
```
