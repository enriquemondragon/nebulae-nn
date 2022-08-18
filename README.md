Nebulae-NN
===============

**Neural network implementation from scratch for celestial objects classification.**


[Charles Messier](https://en.wikipedia.org/wiki/Charles_Messier) was a French astronomer who was interested in finding comets. During his observations, he noticed many non-comet objects, such as nebulae and star clusters, that made his search a little time-consuming. As a solution, he created a list of these objects in order to make his observations more efficient. He titled the list *Catalogue of Nebulae and Star Clusters*, nowadyas we simply call it [Messier objects](https://en.wikipedia.org/wiki/Messier_object).

Motivated to help Messier and implement a neural network from scratch, in this project I implemented a neural network from scratch that would have helped him to classify whether a celestial object is a nebulae or a star cluster (we now know that there are more types of objects).

The model was trained using Hubble's (.jpg) format images available at [NASA](https://www.nasa.gov/content/goddard/hubble-s-messier-catalog#grid)'s webpage. 

![nebulae_nn](/images/nebulae_nn.png)

--------
## Project setup

You can clone this repository by using the command:

```
    $ git clone https://github.com/enriquemondragon/nebulae-nn.git
```
--------
## Usage
To train the model with some default hyperparameters you can just simply enter the folder that contains the images (.png, .jpg, .jpeg) along with their corresponding labels (.csv). Otherwise, you can specify many other things such as the number and dimensions of the layers, activation function for the hidden layers, learning rate and number of epochs (see the --help flag for more info).


```
    python3 nebulae_nn.py --input train_data/ --labels labels.csv 
```
At the end of the training it would save the model (.npy), a figure with the epoch-cost curve, and the cost's history.


To test the model you can just simply enter the folder with the data you want to predict along with the trained model.

```
    python3 nebulae_nn.py --input test_data/ --model model.npy
```
--------
## Status
Currently, nebulae-nn only performs binary classification using binary cross entropy as the loss function and updates the parameters via gradient descent.
In the future, more objective functions can be added, as well as different optimization algorithms.

--------
## Author
Name: Enrique Mondragon Estrada

Mail: emondra99@gmail.com

--------
## Sources
This project was inspired by the specializations:
- NG, A., Mourri, Y. B., Katanforoosh, K. (n.d.). *Deep Learning Specialization*. Coursera. https://www.coursera.org/specializations/deep-learning
- Dye, D., Cooper S. J., Deisenroth M. P., Page, A.F. (n.d.). *Mathematics for Machine Learning Specialization*. Coursera. https://www.coursera.org/specializations/mathematics-machine-learning

--------
## License
Nebulae-NN is available under the MIT license. See the LICENSE file for more info.