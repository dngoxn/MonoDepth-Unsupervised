## Unsupervised Monocular Depth Estimation Model
This is my _PyTorch_ implementation of the paper [Unsupervised Monocular Depth Estimation with Left-Right Consistency](https://arxiv.org/abs/1609.03677) by _Godard et al._ <br>
The authors' original implementation with _TensorFlow_ can be found [here](https://github.com/mrharicot/monodepth).

### Models
I implemented two different models:
- `CNN`: a simple supervised [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network) model used as baseline.
- `Unsupervised_Monocular`: Based on the referenced paper, using [ResNet](https://arxiv.org/abs/1512.03385)-50 as the encoder.

### Data
[NYU-V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) - used for `CNN` model. <br>
[KITTI](https://www.cvlibs.net/datasets/kitti/raw_data.php) - used for `Unsupervised_Monocular` model.

### Note
Due to resource limitations, I used only 10,000 training instances instead of the full dataset ($\sim$ 50,000).
- 7,000 for training
- 1,000 for validation
- 2,000 for testing
<!-- end of the list -->
The model was trained for 50 epochs and took approximately 10 hours. I trained the ResNet encoder from scratch, whereas the authors used a pre-trained ResNet-50.

### Discussion
Sample input-output pairs can be found in the [`output`](./Unsupervised_Monocular/src/output/)directory. Darker colors in the depth maps indicate greater depth.

#### Issues
Some defects were observed in the depth maps:
- **Selective Input**: The model struggles with images containing high levels of detail (e.g., trees, people, shadows), often producing nearly pure noise in these cases.
- **Overfitting**: Depth maps lose coherence after later epochs, settling into local minima that fit the training data well but lack the desired generalization. More details can be found in the [jupyter_notebook](./Unsupervised_Monocular/jupyter_notebook/unsupervised_monocular.ipynb).

#### Positives (and Possible Remedies)
While not perfect, the model shows promise in several areas:
- **Depth Gradient**: Most depth maps exhibit a clear gradient, with reasonable convergence near the horizon (skyline).
- **Object Preservation**: Larger objects are well-represented in the depth maps, though finer details can be improved. The original authors and subsequent models achieved better detail preservation.
- **Baseline Comparison**: Although the CNN baseline is quite simple, comparing outputs highlights the difference between basic recoloring and true depth estimation, reinforcing the potential of unsupervised methods.
