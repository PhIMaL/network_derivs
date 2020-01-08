# Neural Networks with derivatives

Network-derivs allows you to efficiently calculate the derivative of a NN's output w.r.t. to its inputs during the forward pass. That means that you get both the output and its derivatives after a single forward pass. This is especially usefull if you're using PINNs.

**Why?** When using simple autodiff to calculate higher order derivatives, you're calculating many things double (or more). For example, when calculating the third order derivative, autodiff recalculates the first and second order deriv. Since backpropping is very expensive, this is a big part of your total computational effort. 

**How?** If we write a single linear and activation layer as ```f(g(x))```, we can write (for example) the second derivative as ```g'^2 * f + g'' * f'```. By passing the results of lower order derivatives (i.e. f', g') to the next order, we can save a lot of unnecessary calculations.

**What's the speed up?** That depends on the order of your derivs. Assuming every order takes about the same time to calculate (which ofcourse isn't true; higher order derivs are longer calculations), you can expect ~15-25% speedup for second order and 35-45% for third order.  Note that this applies to the time your algorithm spends on these calculations, not your entire algorithm. In my usecase, this is definitely the dominant component so there's a decent speed up.

**What's the effort to implement this in my code?** Almost zero! Network_derivs is basically a drop-in replacement for Pytorch layers (see below). You'll only need to change how to handle the output of your network.

## How to use and implementation
The layers have been designed to be drop-in, i.e. simply change ```torch.nn.``` by ```network_derivs.nn.```. As input to each layer we give a tuple consisting of your standard input tensor and a 'derivative' tensor. Both get propagated through the network and hence accumulate respectively the output and the derivative. The derivative tensor which gets fed into the network accumulates all the (higher order) gradients and has input shape ```(sample, order, feature, feature)```. The first order slice is an identity matrix (as dx/dx=1), while higher order are zero (dx/dx^2... = 0). I've created a function to build this tuple automatically, ```create_deriv_data```.

The network now returns the ```(output, deriv)``` tuple. The deriv part is 4-dimensional tensor with each axis the following meaning:
```(sample, order, input, output)```. So if you want the second derivative of the third output with respect to the first coordinate, simply slice it as:
```deriv[:, 2, 1, 3]```

In the folder 'testing' you'll find a Jupyter notebook I've used to test my code, but it also serves as an example to show how everything works.

## How to install
Simply clone or download the code and run:

```
python setup.py
```

I'm still

## Notes
* So far I've only implemented up to third order derivative and the tanh activation function. I'll probably add support for fourth order soon. Arbitrary order is also possible, but requires solving the Faa di Bruno problem. Right now I don't have any use case for this, but PRs are welcome!

* Right now I also don't calculate mixed derivatives. 

* Suggestions for better names are also welcome :-)
