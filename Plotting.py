from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from sklearn import manifold, datasets
# import plotly.graph_objects as go
# import plotly.io as pio

# pio.orca.config.use_xvfb = True
# print(pio.orca.config)
# print(pio.orca.status)
# pio.orca.config.save()


def show_image(x,output_path):
    plt.imshow(np.squeeze(x),cmap='gray')
    plt.tight_layout()
    plt.savefig(output_path) #end in .png
    
def display_reconstructions(x, x_hat, y, y_hat, output_path):
    w=10
    h=10
    fig=plt.figure(figsize=(16,20))
    plt.title("Inputs and Reconstructions for Two Views: \n" + output_path)
    columns = 4
    rows = 5
    gs = gridspec.GridSpec(rows, columns, hspace=0.05, wspace=0.05)

    for i, g in enumerate(gs):
        col_index = i % columns #the gripspec moves like we read - row by row, left to right
        row_index = i // columns

        if col_index == 0: 
            img = np.squeeze(x[row_index])
        if col_index == 1:
            img = np.squeeze(x_hat[row_index])
        if col_index == 2:
            img = np.squeeze(y[row_index])
        if col_index == 3:
            img = np.squeeze(y_hat[row_index])
        ax = plt.subplot(g)
        ax.imshow(img, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')
        ax.axis('off')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    
    
def display_generated_images(x, y, output_path): 
    w=10
    h=10
    fig=plt.figure(figsize=(10,10))
    plt.title("Randomly Generated Images: \n" + output_path)
    columns = 6
    rows = 6
    gs = gridspec.GridSpec(rows, columns, hspace=0.05, wspace=0.05)

    for i, g in enumerate(gs):
        col_index = i % columns
        row_index = i // columns

        if col_index == 0: 
            img = np.squeeze(x[3*row_index])
        if col_index == 1:
            img = np.squeeze(y[3*row_index])
        if col_index == 2:
            img = np.squeeze(x[3*row_index+1])
        if col_index == 3:
            img = np.squeeze(y[3*row_index+1])
        if col_index == 4:
            img = np.squeeze(x[3*row_index+2])
        if col_index == 5:
            img = np.squeeze(y[3*row_index+2])
        ax = plt.subplot(g)
        ax.imshow(img, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')
        ax.axis('off')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)

def grid_plot2d_private(ae, input_var_name, output_path, output_view_name='x', z=None, hx=None, hy=None):
    w=10
    h=10
    fig=plt.figure(figsize=(14,10))
    cuda = True

    z1 = Variable(torch.from_numpy(np.arange(-3, 3, .25).astype('float32')))
    z2 = Variable(torch.from_numpy(np.arange(-3, 3, .25).astype('float32')))
    if cuda:
        z1, z2 = z1.cuda(), z2.cuda()

    nx, ny = len(z1), len(z2)
    plt.subplot()
    gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)

    for i, g in enumerate(gs):
        index = i // ny
        input_var = torch.cuda.FloatTensor([z1[index], z2[i % nx]]).resize(1, 2)

        if input_var_name == 'z':
            if output_view_name == 'x':
                x_input = torch.cat((input_var+z,hx),1)
                recon = ae.module.decode_x(x_input)
            else:
                y_input = torch.cat((input_var+z,hy),1)
                recon = ae.module.decode_y(y_input)
        elif input_var_name == 'hx':
            x_input = torch.cat((z,input_var+hx),1)
            recon = ae.module.decode_x(x_input)
        elif input_var_name == 'hy':
            y_input = torch.cat((z,input_var+hy),1)
            recon = ae.module.decode_y(y_input)

        ax = plt.subplot(g)
        img = np.array(recon.data.tolist()).reshape(28, 28)
#         ax.imshow(img, cmap='gray')
        ax.imshow(img, )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')
    plt.tight_layout()
    plt.savefig(output_path)
    del fig

def grid_plot2d_single(ae, output_path, output_view_name='x'):
    w=10
    h=10
    fig=plt.figure(figsize=(14,10))
    cuda = True

    z1 = Variable(torch.from_numpy(np.arange(-4, 4, .25).astype('float32')))
    z2 = Variable(torch.from_numpy(np.arange(-4, 4, .25).astype('float32')))
    if cuda:
        z1, z2 = z1.cuda(), z2.cuda()

    nx, ny = len(z1), len(z2)
    plt.subplot()
    gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)

    for i, g in enumerate(gs):
        index = i // ny
        input_var = torch.cuda.FloatTensor([z1[index], z2[i % nx]]).resize(1, 2)

        if output_view_name == 'x':
            recon = ae.module.decode_x(input_var)
        else:
            recon = ae.module.decode_y(input_var)

        ax = plt.subplot(g)
        img = np.array(recon.data.tolist()).reshape(28, 28)
#         ax.imshow(img, cmap='gray')
        ax.imshow(img, )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')
    plt.tight_layout()
    plt.savefig(output_path)
    del fig
    
def plot_embeddings_private(_z, _hx, _hy, labels, output_path):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(20,7))
    fig.suptitle('Q(z|x) Embeddings \n' + output_path)

    ax1.scatter(_z[:,0], _z[:,1], c=labels, alpha=0.5, s=16.)
    ax1.axis('equal')
    ax1.set_xlabel('Z', labelpad = 5)

    ax2.scatter(_hx[:,0], _hx[:,1], c=labels, alpha=0.5, s=16.)
    ax2.axis('equal')
    ax2.set_xlabel('Hx', labelpad = 5)

    ax3.scatter(_hy[:,0], _hy[:,1], c=labels, alpha=0.5, s=16.)
    ax3.axis('equal')
    ax3.set_xlabel('Hy')

    plt.savefig(output_path)

def plot_embeddings_experiment3b(_z, _hx, _hy, labels, rot_x, rot_y, output_path):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(20,7))
    fig.suptitle('Q(z|x) Embeddings \n' + output_path)

    #https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    
    # Compute confusion matrix
    normalize = False
    cm = confusion_matrix(labels,np.argmax(_z,axis=1))
    if normalize:
        cm = np.nan_to_num(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
    classes = np.arange(10)
    print(cm)
    # confusion matrix subplot
    im = ax1.imshow(cm, interpolation='nearest', cmap='RdBu')
    # We want to show all ticks...
    ax1.set(xticks=np.arange(cm.shape[1]),yticks=np.arange(cm.shape[0]),xticklabels=classes, yticklabels=classes, ylabel='Class', xlabel='Z')
    # plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, format(cm[i, j], fmt),ha="center", va="center",color="white" if np.abs(cm[i, j]-thresh) >= .25*cm.max() else "black")
    # ax1.scatter(np.argmax(_z,axis=1), labels, alpha=0.5, s=16.)
    # ax1.axis('equal')
    # ax1.set_xlabel('Z', labelpad = 5)

    ax2.scatter(_hx[:,0], _hx[:,1], c=rot_x, alpha=0.5, s=16.,cmap='Spectral')
    ax2.axis('equal')
    ax2.set_xlabel('Hx', labelpad = 5)

    ax3.scatter(_hy[:,0], _hy[:,1], c=rot_y, alpha=0.5, s=16.,cmap='Spectral')
    ax3.axis('equal')
    ax3.set_xlabel('Hy')

    plt.savefig(output_path)

def save_loss_curves_single(num_epochs, path, accuracies, train_losses={}, val_losses={}):
    fig = plt.figure(figsize=(20,10))
    ax = plt.axes()
    x = np.linspace(0, num_epochs-1, num_epochs)
    train = np.array(train_losses)
    val = np.array(val_losses)
    # print("Loss curve length: ", x.shape)
    colors = {'train':'blue', 'val':'orange', 'accuracies':'green'}
    linestyles = {'total': '-', 'recon': '--', 'discriminative': '-.', 'generative': ':'}
    ax.plot(x, accuracies, color=colors['accuracies'], label="linear svm accuracy")
    for key in train_losses.keys():
        ax.plot(x, train_losses[key], color=colors['train'], linestyle=linestyles[key],label= "train " + key)
    for key in val_losses.keys():
        ax.plot(x, val_losses[key], color=colors['val'], linestyle=linestyles[key], label="val " + key)
    plt.title("Training and Validation Curves: \n" + path)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(path)

# def save_disentangling_curves_private(num_epochs, path, class_accuracies, rot_x_accuracies, rot_y_accuracies, train_losses={}, val_losses={}):
#     fig = plt.figure(figsize=(20,10))
#     ax = plt.axes()
#     x = np.linspace(0, num_epochs-1, num_epochs)
#     train = np.array(train_losses)
#     val = np.array(val_losses)

#     class_accuracies = np.array(class_accuracies)
#     rot_x_accuracies = np.array(rot_x_accuracies)
#     rot_y_accuracies = np.array(rot_y_accuracies)

#     colors = {'train':'blue', 'val':'orange', 'class_accuracies':'green', 'rot_x_accuracies':'cyan', 'rot_y_accuracies':'red'}
#     linestyles = {'z': (0, (5, 1)), 'hx': (0, (1, 1)), 'hy':(0, (3, 1, 1, 1, 1, 1)), 'recon': '-', 'discriminative': '-.', 'generative': ':'}
#     ax.plot(x, class_accuracies[:,0], color=colors['class_accuracies'], linestyle=linestyles['z'], label="z class acc")
#     ax.plot(x, class_accuracies[:,1], color=colors['class_accuracies'], linestyle=linestyles['hx'], label="hx class acc")
#     ax.plot(x, class_accuracies[:,2], color=colors['class_accuracies'], linestyle=linestyles['hy'], label="hy class acc")
#     ax.plot(x, rot_x_accuracies[:,0], color=colors['rot_x_accuracies'], linestyle=linestyles['z'], label="z rot_x acc")
#     ax.plot(x, rot_x_accuracies[:,1], color=colors['rot_x_accuracies'], linestyle=linestyles['hx'], label="hx rot_x acc")
#     ax.plot(x, rot_x_accuracies[:,2], color=colors['rot_x_accuracies'], linestyle=linestyles['hy'], label="hy rot_x acc")
#     ax.plot(x, rot_y_accuracies[:,0], color=colors['rot_y_accuracies'], linestyle=linestyles['z'], label="z rot_y acc")
#     ax.plot(x, rot_y_accuracies[:,1], color=colors['rot_y_accuracies'], linestyle=linestyles['hx'], label="hx rot_y acc")
#     ax.plot(x, rot_y_accuracies[:,2], color=colors['rot_y_accuracies'], linestyle=linestyles['hy'], label="hy rot_y acc")
#     for key in train_losses.keys():
#         ax.plot(x, train_losses[key], color=colors['train'], linestyle=linestyles[key],label= "train " + key)
#     for key in val_losses.keys():
#         ax.plot(x, val_losses[key], color=colors['val'], linestyle=linestyles[key], label="val " + key)
#     plt.title("Training, Validation, and Disentanglement Curves: \n" + path)
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.savefig(path)

def save_disentangling_curves_canonical(num_epochs, path, class_accuracies, rot_x_accuracies, rot_y_accuracies, train_losses={}, val_losses={}):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,16))
    # fig = plt.figure(figsize=(20,10))
    # ax = plt.axes()
    x = np.linspace(0, num_epochs-1, num_epochs)
    train = np.array(train_losses)
    val = np.array(val_losses)

    class_accuracies = np.array(class_accuracies)
    rot_x_accuracies = np.array(rot_x_accuracies)
    rot_y_accuracies = np.array(rot_y_accuracies)

    colors = {'train':'blue', 'val':'orange', 'class_accuracies':'green', 'rot_x_accuracies':'cyan', 'rot_y_accuracies':'red'}
    linestyles = {'z': (0, (5, 1)), 'recon': '-', 'discriminative': '-.', 'generative': ':'}
    ax1.plot(x, class_accuracies[:,0], color=colors['class_accuracies'], linestyle=linestyles['z'], label="z class acc")
    ax1.plot(x, rot_x_accuracies[:,0], color=colors['rot_x_accuracies'], linestyle=linestyles['z'], label="z rot_x acc")
    ax1.plot(x, rot_y_accuracies[:,0], color=colors['rot_y_accuracies'], linestyle=linestyles['z'], label="z rot_y acc")
    ax1.legend(loc="upper right")

    for key in train_losses.keys():
        ax2.plot(x, train_losses[key], color=colors['train'], linestyle=linestyles[key],label= "train " + key)
    for key in val_losses.keys():
        ax2.plot(x, val_losses[key], color=colors['val'], linestyle=linestyles[key], label="val " + key)
    ax2.legend(loc="upper right")

    plt.title("Training, Validation, and Disentanglement Curves: \n" + path)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.legend()
    plt.savefig(path)
    del fig


def plot_embeddings_single(_z, labels, rot_x, rot_y, output_path):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(20,7))
    #https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    classes = np.arange(10)
    # a1 = ax1.scatter(_z[:,0], _z[:,1], c=labels, alpha=0.5, s=16., cmap='jet') #,cmap='jet'
    ax1.scatter(_z[:,0], _z[:,1], c=labels, alpha=0.5, s=16., cmap='jet') #,cmap='jet'
    ax1.axis('equal')
    ax1.set_aspect('equal', 'box')
    ax1.set(xlim=(-4,4), ylim=(-4,4))
    ax1.set_xlabel('Class Coloring', labelpad = 5)
    # fig.colorbar(a1, ax=ax1)
    # a2 = ax2.scatter(_z[:,0], _z[:,1], c=rot_x, alpha=0.5, s=16., cmap='jet')
    ax2.scatter(_z[:,0], _z[:,1], c=rot_x, alpha=0.5, s=16., cmap='jet')
    ax2.axis('equal')
    ax2.set_aspect('equal', 'box')
    ax2.set(xlim=(-4,4), ylim=(-4,4))
    ax2.set_xlabel('X Rotation Angle Coloring', labelpad = 5)
    # fig.colorbar(a2, ax=ax2)
    # a3 = ax3.scatter(_z[:,0], _z[:,1], c=rot_y, alpha=0.5, s=16., cmap='jet')
    ax3.scatter(_z[:,0], _z[:,1], c=rot_y, alpha=0.5, s=16., cmap='jet')
    ax3.axis('equal')
    ax3.set_aspect('equal', 'box')
    ax3.set(xlim=(-4,4), ylim=(-4,4))
    ax3.set_xlabel('Y Rotation Angle Coloring')
    # fig.colorbar(a3, ax=ax3)
    plt.savefig(output_path)
    del fig

def plot_contours_single_(_z, output_path):
    x = np.linspace(-4, 4, 80)
    y = np.linspace(-4, 4, 80)
    X, Y = np.meshgrid(x, y)

    # build a density model using KDE
    
    Z = f(X, Y)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(20,7))
    #https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    classes = np.arange(10)
    ax1.scatter(_z[:,0], _z[:,1], c=labels, alpha=0.5, s=16., cmap='jet') #,cmap='jet'
    ax1.axis('equal')
    ax1.set_xlabel('Class Coloring', labelpad = 5)
    ax2.scatter(_z[:,0], _z[:,1], c=rot_x, alpha=0.5, s=16., cmap='jet')
    ax2.axis('equal')
    ax2.set_xlabel('X Rotation Angle Coloring', labelpad = 5)
    ax3.scatter(_z[:,0], _z[:,1], c=rot_y, alpha=0.5, s=16., cmap='jet')
    ax3.axis('equal')
    ax3.set_xlabel('Y Rotation Angle Coloring')
    plt.savefig(output_path)
    del fig

def save_disentangling_curves_single(results, output_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,16))
    
    num_epochs = len(results.training_re_losses)
    x = np.linspace(0, num_epochs-1, num_epochs)

    colors = {'train':'blue', 'val':'orange', 'class_accuracies':'green', 'rot_x_accuracies':'cyan', 'rot_y_accuracies':'red'}
    linestyles = {'z': (0, (5, 1)), 'recon': '-', 'discriminative': '-.', 'generative': ':'}
    ax1.plot(x, np.array(results.class_predictions)[:,0], color=colors['class_accuracies'], linestyle=linestyles['z'], label="z class acc")
    ax1.plot(x, np.array(results.rotx_predictions)[:,0], color=colors['rot_x_accuracies'], linestyle=linestyles['z'], label="z rot_x acc")
    ax1.plot(x, np.array(results.roty_predictions)[:,0], color=colors['rot_y_accuracies'], linestyle=linestyles['z'], label="z rot_y acc")
    ax1.legend(loc="upper right")

    if len(results.training_g_losses) > 0:
        ax2.plot(x, np.array(results.training_g_losses['z']), color=colors['train'], linestyle=linestyles['generative'],label= "train generative")
    if len(results.training_d_losses) > 0:
        ax2.plot(x, np.array(results.training_d_losses['z']), color=colors['train'], linestyle=linestyles['discriminative'],label= "train discriminative")
    ax2.plot(x, np.array(results.training_re_losses), color=colors['train'], linestyle=linestyles['recon'],label= "train recon")
    ax2.legend(loc="upper right")

    plt.title("Training and Prediction Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.legend()
    plt.savefig(output_path)

def save_disentangling_curves_private(results, output_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,16))
    
    num_epochs = len(results.training_re_losses)
    x = np.linspace(0, num_epochs-1, num_epochs)

    colors = {'train':'blue', 'class_accuracies':'green', 'rot_x_accuracies':'cyan', 'rot_y_accuracies':'red', 'discriminative': 'blue', 'generative': 'green', 'recon': 'red'}
    linestyles = {'z': (0, (5, 1)), 'hx': (0, (1, 1)), 'hy':(0, (3, 1, 1, 1, 1, 1))}
    ax1.plot(x, np.array(results.class_predictions)[:,0], color=colors['class_accuracies'], linestyle=linestyles['z'], label="z class acc")
    ax1.plot(x, np.array(results.class_predictions)[:,1], color=colors['class_accuracies'], linestyle=linestyles['hx'], label="hx class acc")
    ax1.plot(x, np.array(results.class_predictions)[:,2], color=colors['class_accuracies'], linestyle=linestyles['hy'], label="hy class acc")
    ax1.plot(x, np.array(results.rotx_predictions)[:,0], color=colors['rot_x_accuracies'], linestyle=linestyles['z'], label="z rot_x acc")
    ax1.plot(x, np.array(results.rotx_predictions)[:,1], color=colors['rot_x_accuracies'], linestyle=linestyles['hx'], label="hx rot_x acc")
    ax1.plot(x, np.array(results.rotx_predictions)[:,2], color=colors['rot_x_accuracies'], linestyle=linestyles['hy'], label="hy rot_x acc")
    ax1.plot(x, np.array(results.roty_predictions)[:,0], color=colors['rot_y_accuracies'], linestyle=linestyles['z'], label="z rot_y acc")
    ax1.plot(x, np.array(results.roty_predictions)[:,1], color=colors['rot_y_accuracies'], linestyle=linestyles['hx'], label="hx rot_y acc")
    ax1.plot(x, np.array(results.roty_predictions)[:,2], color=colors['rot_y_accuracies'], linestyle=linestyles['hy'], label="hy rot_y acc")
    ax1.legend(loc="upper right")

    for g_loss in results.training_g_losses.keys():
        ax2.plot(x, np.array(results.training_g_losses[g_loss]), color=colors['generative'], linestyle=linestyles[g_loss], label=g_loss + "_generative")
    for d_loss in results.training_d_losses.keys():
        ax2.plot(x, np.array(results.training_d_losses[d_loss]), color=colors['discriminative'], linestyle=linestyles[d_loss], label=d_loss + "_discriminative")
    ax2.plot(x, np.array(results.training_re_losses), color=colors['recon'], label= "train recon")
    ax2.legend(loc="upper right")

    plt.title("Training and Prediction Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.legend()
    plt.savefig(output_path)

def plot_grid_generations_single(ae, x_output_path, y_output_path):
    # 2d Grid Generations
    zero_tensor = torch.zeros([1, 2], dtype=torch.float32, device="cuda")

    # view x
    print('Generating 2d grid plots using z for view x')
    grid_plot_path = best_ae_path.replace('.pt','_z_viewx_gridgenerations.png')
    if os.path.exists(grid_plot_path):
        os.remove(grid_plot_path)
    grid_plot2d(ae, input_var_name='z', output_path=grid_plot_path, output_view_name='x', z=zero_tensor, hx=zero_tensor)
    
    # view y
    print('Generating 2d grid plots using z for view y')
    grid_plot_path = best_ae_path.replace('.pt','_z_viewy_gridgenerations.png')
    if os.path.exists(grid_plot_path):
        os.remove(grid_plot_path)
    grid_plot2d(ae, input_var_name='z', output_path=grid_plot_path, output_view_name='y', z=zero_tensor, hy=zero_tensor)

def plot_3d_embeddings(z, class_info, file_path=None, cmap_name='Spectral', plot_dataset=False):
    if plot_dataset:
        z, class_info = datasets.samples_generator.make_s_curve(10000) 
    
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111, projection='3d')

    if plot_dataset:
        ax.scatter(z[:,0], z[:,1], z[:,2], c=class_info, alpha=0.5)
    else:
        ax.scatter(z[:,0], z[:,1], z[:,2], c=class_info, alpha=0.5,cmap=cmap_name)
    # ax.axis('equal')
    # ax.set_xlabel('Z', labelpad = 5)
    ax.view_init(4, -72)    
    
    if file_path is not None:
        plt.savefig(file_path, bbox_inches = 'tight', pad_inches = 0.05)
    else:
        plt.show()
        
def plot_s_curve_embeddings(z, hx, hy, class_info, rot_x, rot_y, file_path=None, cmap_name='Spectral', plot_dataset=False):
    fig = plt.figure(figsize=(20,6))
    
    # plot 2d hx
    ax = fig.add_subplot(1, 3, 1)
    ax.scatter(hx[:,0], hx[:,1], c=rot_x, alpha=0.5, s=16.,cmap=cmap_name)
    ax.axis('equal')
    ax.set_xlabel('Hx', labelpad = 5)
    
    # plot 3d z
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.scatter(z[:,0], z[:,1], z[:,2], c=class_info, alpha=0.5,s=1., cmap=cmap_name)
    ax.axis('equal')
    ax.set_xlabel('Z', labelpad = 5)
    ax.view_init(4, -72) 
    
    # plot 2d hy
    ax = fig.add_subplot(1, 3, 3)
    ax.scatter(hy[:,0], hy[:,1], c=rot_y, alpha=0.5, s=16.,cmap=cmap_name)
    ax.axis('equal')
    ax.set_xlabel('Hy', labelpad = 5)

    if file_path is not None:
        plt.savefig(file_path, bbox_inches = 'tight', pad_inches = 0.05)
    else:
        plt.show()

# def plotly_3d_scatter(z, class_info, file_path=None, plot_dataset=False):
#     if plot_dataset:
#         z, class_info = datasets.samples_generator.make_s_curve(10000) 

#     fig = go.Figure(data=[go.Scatter3d(
#         x=z[0:10000,0],
#         y=z[0:10000,1],
#         z=z[0:10000,2],
#         mode='markers',
#         marker=dict(
#             size=1,
#             color=class_info,                # set color to an array/list of desired values
#             colorscale='Viridis',   # choose a colorscale
#             opacity=0.8
#         )
#     )])

#     # tight layout
#     fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
#     # fig.show()
#     fig.write_image(file_path)

def display_datasets(x_noisy, y_noisy, x_tangled, y_tangled, output_path):
    w=10
    h=10
    fig=plt.figure(figsize=(60,14))
    # plt.title("Inputs and Reconstructions for Two Views: \n" + output_path)
    columns = 15
    rows = 4
    gs = gridspec.GridSpec(rows, columns, hspace=0.05, wspace=0.05)

    for i, g in enumerate(gs):
        #the gripspec moves like we read - row by row, left to right
        col_index = i % columns 
        row_index = i // columns

        if row_index == 0: 
            img = np.squeeze(x_noisy[col_index])
        if row_index == 1:
            img = np.squeeze(y_noisy[col_index])
        if row_index == 2:
            img = np.squeeze(x_tangled[col_index])
        if row_index == 3:
            img = np.squeeze(y_tangled[col_index])
        ax = plt.subplot(g)
        ax.imshow(img, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')
        ax.axis('off')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)