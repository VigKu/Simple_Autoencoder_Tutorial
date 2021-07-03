import matplotlib.pyplot as plt


def plot_images(outputs, num_epochs):
    for k in range(0, num_epochs, 2):
        plt.figure(figsize=(9, 2))
        plt.gray()
        data = outputs[k][1].detach().numpy()
        rebuilt_data = outputs[k][2].detach().numpy()
        for i, item in enumerate(data):
            if i >= 9:
                break
            plt.subplot(2, 9, i + 1)
            plt.imshow(item[0])

        for i, item in enumerate(rebuilt_data):
            if i >= 9:
                break
            plt.subplot(2, 9, 9 + i + 1)
            plt.imshow(item[0])
