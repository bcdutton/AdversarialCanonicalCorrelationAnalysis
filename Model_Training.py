from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from datetime import datetime
import os
from os import path
from tempfile import TemporaryFile
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import pycuda
from pycuda import compiler
import pycuda.driver as drv
from sklearn.svm import LinearSVC, LinearSVR
from sklearn import manifold, datasets
import math

class TrainingResults:
    def __init__(self):
        self.training_g_losses = {}
        self.val_g_losses = []
        self.training_d_losses = {}
        self.val_d_losses = []
        self.training_re_losses = []
        self.validation_losses = []
        self.class_predictions = []
        self.rotx_predictions = []
        self.roty_predictions = []


def vcca_single_loss(x_hat, x, y_hat, y, z_mu, z_logvar, beta=1.0):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    re_loss = nn.BCELoss(reduction='sum')
    x_BCE = re_loss(x_hat, x)
    y_BCE = re_loss(y_hat, y)
    z_KLD = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    return x_BCE + y_BCE + (beta * z_KLD)


def vcca_private_loss(x_hat, x, y_hat, y, z_mu, z_logvar, hx_mu, hx_logvar, hy_mu, hy_logvar, alpha=1.0, beta=1.0, gamma=1.0):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    re_loss = nn.BCELoss(reduction='sum')
    x_BCE = re_loss(x_hat, x)
    y_BCE = re_loss(y_hat, y)
    z_KLD = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    hx_KLD = -0.5 * torch.sum(1 + hx_logvar - hx_mu.pow(2) - hx_logvar.exp())
    hy_KLD = -0.5 * torch.sum(1 + hy_logvar - hy_mu.pow(2) - hy_logvar.exp())
    return x_BCE + y_BCE + (beta * z_KLD) + (alpha * hx_KLD) + (gamma * hy_KLD)


def train_acca_single(ae, z_dim, discriminator, train_loader, validation_loader, test_loader, num_epochs, recon_loss, device, prior='gaussian'): # best_ae_path=None, best_discriminator_path=None
    if prior == 'S_manifold':
        s_curve, _ = datasets.samples_generator.make_s_curve(60000,noise=0.05)

    adversarial_loss = torch.nn.BCELoss()
    if recon_loss== 'L1':
        pixelwise_loss = nn.L1Loss()
    elif recon_loss=='MSE' or recon_loss=='L2':
        pixelwise_loss = nn.MSELoss()
    elif recon_loss=='BCE':
        pixelwise_loss = nn.BCELoss()
        
    learning_rate = 1e-4
    optimizer_Gz = torch.optim.Adam(ae.module.encode_z.parameters(), lr=learning_rate)
    optimizer_Dz = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
    optimizer_AE = torch.optim.Adam(ae.parameters(), lr=learning_rate)

    best_accuracy = 0.0
    best_epoch = 0
    best_loss = np.inf

    results = TrainingResults()
    results.training_g_losses['z'] = []
    results.training_d_losses['z'] = []

    best_state_dict = ae.state_dict()
    best_acc_state_dict = ae.state_dict()
    models = []

    for j in range(num_epochs):
        running_g_loss = 0.0
        running_d_loss = 0.0
        running_re_loss = 0.0
        
        ae.train()
        for i, data in enumerate(train_loader, 0): 
            x, y = data[0].to(device), data[1].to(device)

            # Adversarial ground truths
            valid = Variable(torch.cuda.FloatTensor(x.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(torch.cuda.FloatTensor(x.shape[0], 1).fill_(0.0), requires_grad=False)

            # ----------------
            #  Train Reconstruction
            # ----------------
            optimizer_AE.zero_grad()

            z = ae.module.encode(x,y)
            x_hat, y_hat = ae.module.decode(z)
            
            re_loss = pixelwise_loss(x_hat, x) + pixelwise_loss(y_hat, y)
            re_loss.backward()
            optimizer_AE.step()

            # -----------------
            #  Train Generator 
            # -----------------
            optimizer_Gz.zero_grad()

            # Loss measures generator's ability to fool the discriminator
            z = ae.module.encode(x,y)
            gz_loss = adversarial_loss(discriminator(z), Variable(torch.cuda.FloatTensor(x.shape[0], 1).fill_(0.0), requires_grad=False))
            gz_loss.backward()

            optimizer_Gz.step()

            # ---------------------
            #  Train Discriminator 
            # ---------------------
            optimizer_Dz.zero_grad()

            # Sample noise as discriminator ground truth
            if prior == 'gaussian':
                fake_z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (x.shape[0], z_dim))))
            elif prior == 'S_manifold':
                idx = np.random.randint(60000, size=x.shape[0])
                fake_z = Variable(torch.cuda.FloatTensor(s_curve[idx,:]))

            # Measure discriminator's ability to classify real from generated samples
            real_z_loss = adversarial_loss(discriminator(fake_z), fake)

            z = ae.module.encode(x, y)
            fake_z_loss = adversarial_loss(discriminator(z), valid)
            
            d_z_loss = 0.5 * (real_z_loss + fake_z_loss)

            d_z_loss.backward()

            optimizer_Dz.step()

            running_g_loss += gz_loss.item()
            running_d_loss += d_z_loss.item()
            running_re_loss += re_loss.item()

        ae.eval()

        re_loss = running_re_loss/(i+1)
        g_loss = running_g_loss/(i+1)
        d_loss = running_d_loss/(i+1)
        val_g_loss = abs(-math.log(0.5) - g_loss)
        val_d_loss = abs(-math.log(0.5) - d_loss)

        results.training_g_losses['z'].append(g_loss)
        results.training_d_losses['z'].append(d_loss)
        results.training_re_losses.append(re_loss)
        print(j, "TRAINING g_loss: ", g_loss, " d_loss: ", d_loss, "re_loss: ", re_loss)

        # Train classifier
        x, y, rot_x, rot_y, labels = next(iter(validation_loader))
        z = ae.module.encode(x.to(device),y.to(device))
        train_z, train_labels, train_rotx, train_roty = z.detach().cpu().numpy(), labels.cpu().numpy(), rot_x.cpu().numpy(), rot_y.cpu().numpy()
        clf, regr_x, regr_y = [], [], []
        inputs = [train_z]
        for i in range(len(inputs)):
            clf.append(LinearSVC(random_state=0, tol=1e-5))
            regr_x.append(LinearSVR(random_state=0, tol=1e-5))
            regr_y.append(LinearSVR(random_state=0, tol=1e-5))
        for i in range(len(inputs)):
            clf[i].fit(inputs[i], train_labels)
            regr_x[i].fit(inputs[i], train_rotx)
            regr_y[i].fit(inputs[i], train_roty)
        #evaluate
        x, y, rot_x, rot_y, labels = next(iter(test_loader))
        z = ae.module.encode(x.to(device),y.to(device))
        test_z, test_labels, test_rotx, test_roty = z.detach().cpu().numpy(), labels.cpu().numpy(), rot_x.cpu().numpy(), rot_y.cpu().numpy()
        inputs = [test_z]

        class_scores, rot_x_scores, rot_y_scores = [], [], []
        for i in range(len(inputs)):
            class_scores.append(clf[i].score(inputs[i], test_labels))
            rot_x_scores.append(regr_x[i].score(inputs[i], test_rotx))
            rot_y_scores.append(regr_y[i].score(inputs[i], test_roty))
        accuracy = .3333 * (class_scores[0] + rot_x_scores[0] + rot_y_scores[0])
        print('accuracy: ', accuracy)

        results.class_predictions.append(class_scores)
        results.rotx_predictions.append(rot_x_scores)
        results.roty_predictions.append(rot_y_scores)
    
        # Saving Loss curves
        if j % 10 == 0:
            models.append(ae.state_dict())

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_acc_state_dict = ae.state_dict()

        total_loss = val_g_loss + val_d_loss + re_loss
        if total_loss < best_loss:
            print("new best loss", total_loss)
            best_state_dict = ae.state_dict()
            best_loss = total_loss
            best_epoch = j
            
    return best_accuracy, best_epoch, best_state_dict, results, ae.state_dict(), best_acc_state_dict, models


def train_acca_private(ae, z_dim, hx_dim, hy_dim, discriminators, train_loader, validation_loader, test_loader, num_epochs, recon_loss, device):

    adversarial_loss = torch.nn.BCELoss()
    if recon_loss== 'L1':
        pixelwise_loss = nn.L1Loss()
    elif recon_loss=='MSE' or recon_loss=='L2':
        pixelwise_loss = nn.MSELoss()
    elif recon_loss=='BCE':
        pixelwise_loss = nn.BCELoss()
        
    learning_rate = 1e-4
    optimizer_Gz = torch.optim.Adam(ae.module.encode_z.parameters(), lr=learning_rate)
    optimizer_Gx = torch.optim.Adam(ae.module.encode_hx.parameters(), lr=learning_rate)
    optimizer_Gy = torch.optim.Adam(ae.module.encode_hy.parameters(), lr=learning_rate)
    optimizer_Dz = torch.optim.Adam(discriminators[0].parameters(), lr=learning_rate)
    optimizer_Dx = torch.optim.Adam(discriminators[1].parameters(), lr=learning_rate)
    optimizer_Dy = torch.optim.Adam(discriminators[2].parameters(), lr=learning_rate)
    optimizer_AE = torch.optim.Adam(ae.parameters(), lr=learning_rate)

    best_accuracy = 0.0
    best_epoch = 0
    best_loss = np.inf

    results = TrainingResults()
    results.training_g_losses['z'] = []
    results.training_g_losses['hx'] = []
    results.training_g_losses['hy'] = []
    results.training_d_losses['z'] = []
    results.training_d_losses['hx'] = []
    results.training_d_losses['hy'] = []

    best_state_dict = ae.state_dict()
    best_acc_state_dict = ae.state_dict()
    models = []

    for j in range(num_epochs):
        running_g_loss = 0.0
        running_d_loss = 0.0
        running_gz_loss = 0.0
        running_gx_loss = 0.0
        running_gy_loss = 0.0
        running_dz_loss = 0.0
        running_dx_loss = 0.0
        running_dy_loss = 0.0
        running_val_g_loss = 0.0
        running_val_d_loss = 0.0
        running_re_loss = 0.0
        
        ae.train()
        for i, data in enumerate(train_loader, 0): 
            x, y = data[0].to(device), data[1].to(device)

            # Adversarial ground truths
            valid = Variable(torch.cuda.FloatTensor(x.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(torch.cuda.FloatTensor(x.shape[0], 1).fill_(0.0), requires_grad=False)

            # ----------------
            #  Train Reconstruction
            # ----------------
            optimizer_AE.zero_grad()

            z, hx, hy = ae.module.encode(x,y)
            x_hat, y_hat = ae.module.decode(z, hx, hy)
            
            re_loss = pixelwise_loss(x_hat, x) + pixelwise_loss(y_hat, y)
            re_loss.backward()
            optimizer_AE.step()

            # -----------------
            #  Train Generator 
            # -----------------
            optimizer_Gz.zero_grad()
            optimizer_Gx.zero_grad()
            optimizer_Gy.zero_grad()

            # Loss measures generator's ability to fool the discriminator
            z, hx, hy = ae.module.encode(x,y)
            gz_loss = adversarial_loss(discriminators[0](z), Variable(torch.cuda.FloatTensor(x.shape[0], 1).fill_(0.0), requires_grad=False))
            gx_loss = adversarial_loss(discriminators[1](hx), Variable(torch.cuda.FloatTensor(x.shape[0], 1).fill_(0.0), requires_grad=False))
            gy_loss = adversarial_loss(discriminators[2](hy), Variable(torch.cuda.FloatTensor(x.shape[0], 1).fill_(0.0), requires_grad=False))

            gz_loss.backward()
            gx_loss.backward()
            gy_loss.backward()

            optimizer_Gz.step()
            optimizer_Gx.step()
            optimizer_Gy.step()

            # ---------------------
            #  Train Discriminator 
            # ---------------------
            optimizer_Dz.zero_grad()
            optimizer_Dx.zero_grad()
            optimizer_Dy.zero_grad()

            # Sample noise as discriminator ground truth
            fake_z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (x.shape[0], z_dim))))
            fake_hx = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (x.shape[0], hx_dim))))
            fake_hy = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (x.shape[0], hy_dim))))

            # Measure discriminator's ability to classify real from generated samples
            real_z_loss = adversarial_loss(discriminators[0](fake_z), fake)
            real_hx_loss = adversarial_loss(discriminators[1](fake_hx), fake)
            real_hy_loss = adversarial_loss(discriminators[2](fake_hy), fake)

            z, hx, hy = ae.module.encode(x, y)
            fake_z_loss = adversarial_loss(discriminators[0](z), valid)
            fake_hx_loss = adversarial_loss(discriminators[1](hx), valid)
            fake_hy_loss = adversarial_loss(discriminators[2](hy), valid)
            
            d_z_loss = 0.5 * (real_z_loss + fake_z_loss)
            d_hx_loss = 0.5 * (real_hx_loss + fake_hx_loss)
            d_hy_loss = 0.5 * (real_hy_loss + fake_hy_loss)

            d_z_loss.backward()
            d_hx_loss.backward()
            d_hy_loss.backward()

            optimizer_Dz.step()
            optimizer_Dx.step()
            optimizer_Dy.step()

            running_g_loss += .3333 * (gz_loss.item() + gx_loss.item() + gy_loss.item())
            running_d_loss += .3333 * (d_z_loss.item() + d_hx_loss.item() + d_hy_loss.item())
            running_gz_loss += gz_loss.item()
            running_gx_loss += gx_loss.item()
            running_gy_loss += gy_loss.item()
            running_dz_loss += d_z_loss.item()
            running_dx_loss += d_hx_loss.item()
            running_dy_loss += d_hy_loss.item()
            running_re_loss += re_loss.item()

            if i==0 and j==0:
                print("Initial values (gen, disc, re): ", running_g_loss, running_d_loss, running_re_loss)

        ae.eval()

        gx_loss = running_gx_loss/(i+1)
        gy_loss = running_gy_loss/(i+1)
        gz_loss = running_gz_loss/(i+1)
        dx_loss = running_dx_loss/(i+1)
        dy_loss = running_dy_loss/(i+1)
        dz_loss = running_dz_loss/(i+1)
        re_loss = running_re_loss/(i+1)
        g_loss = running_g_loss/(i+1)
        d_loss = running_d_loss/(i+1)
        val_g_loss = (abs(-math.log(0.5)-gz_loss) + abs(-math.log(0.5)-gx_loss) + abs(-math.log(0.5)-gy_loss)) / 3.0
        val_d_loss = (abs(-math.log(0.5)-dz_loss) + abs(-math.log(0.5)-dx_loss) + abs(-math.log(0.5)-dy_loss)) / 3.0

        # results.training_g_losses.append(g_loss)
        # results.training_d_losses.append(d_loss)
        results.training_g_losses['z'].append(running_gz_loss/(i+1))
        results.training_g_losses['hx'].append(running_gx_loss/(i+1))
        results.training_g_losses['hy'].append(running_gy_loss/(i+1))
        results.training_d_losses['z'].append(running_dz_loss/(i+1))
        results.training_d_losses['hx'].append(running_dx_loss/(i+1))
        results.training_d_losses['hy'].append(running_dy_loss/(i+1))
        results.training_re_losses.append(re_loss)
        results.val_g_losses.append(val_g_loss)
        results.val_d_losses.append(val_d_loss)
        print(j, "TRAINING losses... g:", g_loss, " d:", d_loss, "re:", re_loss, "val_g:", val_g_loss, "val_d:", val_d_loss)

        # Train classifier
        x, y, rot_x, rot_y, labels = next(iter(validation_loader))
        z, hx, hy = ae.module.encode(x.to(device),y.to(device))
        train_z, train_hx, train_hy, train_labels, train_rotx, train_roty = z.detach().cpu().numpy(), hx.detach().cpu().numpy(), hy.detach().cpu().numpy(), labels.cpu().numpy(), rot_x.cpu().numpy(), rot_y.cpu().numpy()
        clf, regr_x, regr_y = [], [], []
        inputs = [train_z, train_hx, train_hy]
        for i in range(3):
            clf.append(LinearSVC(random_state=0, tol=1e-5))
            regr_x.append(LinearSVR(random_state=0, tol=1e-5))
            regr_y.append(LinearSVR(random_state=0, tol=1e-5))
        for i in range(3):
            clf[i].fit(inputs[i], train_labels)
            regr_x[i].fit(inputs[i], train_rotx)
            regr_y[i].fit(inputs[i], train_roty)
       
        #evaluate
        x, y, rot_x, rot_y, labels = next(iter(test_loader))
        z, hx, hy = ae.module.encode(x.to(device),y.to(device))
        test_z, test_hx, test_hy, test_labels, test_rotx, test_roty = z.detach().cpu().numpy(), hx.detach().cpu().numpy(), hy.detach().cpu().numpy(), labels.cpu().numpy(), rot_x.cpu().numpy(), rot_y.cpu().numpy()
        inputs = [test_z, test_hx, test_hy]
        class_scores, rot_x_scores, rot_y_scores = [], [], []
        for i in range(3):
            class_scores.append(clf[i].score(inputs[i], test_labels))
            rot_x_scores.append(regr_x[i].score(inputs[i], test_rotx))
            rot_y_scores.append(regr_y[i].score(inputs[i], test_roty))
        accuracy = .3333 * (class_scores[0] + rot_x_scores[1] + rot_y_scores[2])
        print('accuracy: ', accuracy)

        results.class_predictions.append(class_scores)
        results.rotx_predictions.append(rot_x_scores)
        results.roty_predictions.append(rot_y_scores)
    
        # Saving Loss curves
        if j % 10 == 0:
            models.append(ae.state_dict())

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_acc_state_dict = ae.state_dict()

        total_loss = val_g_loss + val_d_loss + re_loss
        if total_loss < best_loss:
            print("new best loss", total_loss)
            best_state_dict = ae.state_dict()
            best_loss = total_loss
            best_epoch = j
    
    return best_accuracy, best_epoch, best_state_dict, results, ae.state_dict(), best_acc_state_dict, models


def train_vcca_single(ae, z_dim, train_loader, validation_loader, test_loader, num_epochs, device):
    optimizer_AE = torch.optim.Adam(ae.parameters(), lr=1e-3)

    best_accuracy = 0.0
    best_epoch = 0
    best_loss = np.inf

    results = TrainingResults()

    best_state_dict = ae.state_dict()
    best_acc_state_dict = ae.state_dict()
    models = []

    for j in range(num_epochs):
        running_re_loss = 0.0
        
        ae.train()
        for i, data in enumerate(train_loader, 0): 
            x, y = data[0].to(device), data[1].to(device)

            optimizer_AE.zero_grad()

            x_hat, y_hat, z_mu, z_logvar, z = ae(x,y)
            
            loss = vcca_single_loss(x_hat, x, y_hat, y, z_mu, z_logvar)
            loss.backward()
            optimizer_AE.step()

            running_re_loss += loss.item()

        ae.eval()
        re_loss = running_re_loss/(i+1)
        
        results.training_re_losses.append(re_loss)
        
        print(j, "TRAINING loss: ", re_loss)

        # Train classifier
        x, y, rot_x, rot_y, labels = next(iter(validation_loader))
        x_hat, y_hat, z_mu, z_logvar, z = ae(x.to(device), y.to(device))
        train_z, train_labels, train_rot_x, train_rot_y = z_mu.detach().cpu().numpy(), labels.cpu().numpy(), rot_x.cpu().numpy(), rot_y.cpu().numpy()
        clf, regr_a, regr_b = [], [], []
        inputs = [train_z]
        for i in range(len(inputs)):
            clf.append(LinearSVC(random_state=0, tol=1e-5))
            regr_a.append(LinearSVR(random_state=0, tol=1e-5))
            regr_b.append(LinearSVR(random_state=0, tol=1e-5))
        for i in range(len(inputs)):
            clf[i].fit(inputs[i], train_labels)
            regr_a[i].fit(inputs[i], train_rot_x)
            regr_b[i].fit(inputs[i], train_rot_y)
        #evaluate
        x, y, rot_x, rot_y, labels = next(iter(test_loader))
        x_hat, y_hat, z_mu, z_logvar, z = ae(x.to(device),y.to(device))
        test_z, test_labels, test_rot_x, test_rot_y = z_mu.detach().cpu().numpy(), labels.cpu().numpy(), rot_x.cpu().numpy(), rot_y.cpu().numpy()
        inputs = [test_z]

        class_scores, rot_x_scores, rot_y_scores = [], [], []
        for i in range(len(inputs)):
            class_scores.append(clf[i].score(inputs[i], test_labels))
            rot_x_scores.append(regr_a[i].score(inputs[i], test_rot_x))
            rot_y_scores.append(regr_b[i].score(inputs[i], test_rot_y))
        accuracy = .3333 * (class_scores[0] + rot_x_scores[0] + rot_y_scores[0])
        print('accuracy: ', accuracy)

        results.class_predictions.append(class_scores)
        results.rotx_predictions.append(rot_x_scores)
        results.roty_predictions.append(rot_y_scores)

        # Saving Loss curves
        if j % 10 == 0:
            models.append(ae.state_dict())

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_acc_state_dict = ae.state_dict()

        if re_loss < best_loss:
            print("new best loss", re_loss)
            best_state_dict = ae.state_dict()
            best_loss = re_loss
            best_epoch = j

    return best_accuracy, best_epoch, best_state_dict, results, ae.state_dict(), best_acc_state_dict, models


def train_vcca_private(ae, z_dim, hx_dim, hy_dim, train_loader, validation_loader, test_loader, num_epochs, device):    
    optimizer_AE = torch.optim.Adam(ae.parameters(), lr=1e-3)

    best_accuracy = 0.0
    best_epoch = 0
    best_loss = np.inf

    results = TrainingResults()

    best_state_dict = ae.state_dict()
    best_acc_state_dict = ae.state_dict()
    models = []

    for j in range(num_epochs):
        running_re_loss = 0.0
        
        ae.train()
        for i, data in enumerate(train_loader, 0): 
            x, y = data[0].to(device), data[1].to(device)

            optimizer_AE.zero_grad()
            
            x_hat, y_hat, z_mu, z_logvar, hx_mu, hx_logvar, hy_mu, hy_logvar, z, hx, hy = ae(x,y)
            
            loss = vcca_private_loss(x_hat, x, y_hat, y, z_mu, z_logvar, hx_mu, hx_logvar, hy_mu, hy_logvar)
            loss.backward()
            optimizer_AE.step()

            running_re_loss += loss.item()

        ae.eval()
        re_loss = running_re_loss/(i+1)
        
        results.training_re_losses.append(re_loss)
        
        print(j, "TRAINING loss: ", re_loss)

        # Train classifier
        x, y, rot_x, rot_y, labels = next(iter(validation_loader))
        x_hat, y_hat, z_mu, z_logvar, hx_mu, hx_logvar, hy_mu, hy_logvar, z, hx, hy = ae(x.to(device),y.to(device))
        train_z, train_hx, train_hy, train_labels, train_rot_x, train_rot_y = z_mu.detach().cpu().numpy(), hx_mu.detach().cpu().numpy(), hy_mu.detach().cpu().numpy(), labels.cpu().numpy(), rot_x.cpu().numpy(), rot_y.cpu().numpy()
        clf, regr_a, regr_b = [], [], []
        inputs = [train_z, train_hx, train_hy]
        for i in range(3):
            clf.append(LinearSVC(random_state=0, tol=1e-5))
            regr_a.append(LinearSVR(random_state=0, tol=1e-5))
            regr_b.append(LinearSVR(random_state=0, tol=1e-5))
        for i in range(3):
            clf[i].fit(inputs[i], train_labels)
            regr_a[i].fit(inputs[i], train_rot_x)
            regr_b[i].fit(inputs[i], train_rot_y)
        #evaluate
        x, y, rot_x, rot_y, labels = next(iter(test_loader))
        x_hat, y_hat, z_mu, z_logvar, hx_mu, hx_logvar, hy_mu, hy_logvar, z, hx, hy = ae(x.to(device),y.to(device))
        test_z, test_hx, test_hy, test_labels, test_rot_x, test_rot_y = z_mu.detach().cpu().numpy(), hx_mu.detach().cpu().numpy(), hy_mu.detach().cpu().numpy(), labels.cpu().numpy(), rot_x.cpu().numpy(), rot_y.cpu().numpy()
        inputs = [test_z, test_hx, test_hy]

        class_scores, rot_x_scores, rot_y_scores = [], [], []
        for i in range(3):
            class_scores.append(clf[i].score(inputs[i], test_labels))
            rot_x_scores.append(regr_a[i].score(inputs[i], test_rot_x))
            rot_y_scores.append(regr_b[i].score(inputs[i], test_rot_y))
        accuracy = .3333 * (class_scores[0] + rot_x_scores[1] + rot_y_scores[2])
        print('accuracy: ', accuracy)

        results.class_predictions.append(class_scores)
        results.rotx_predictions.append(rot_x_scores)
        results.roty_predictions.append(rot_y_scores)

        # Saving loss curves
        if j % 10 == 0:
            models.append(ae.state_dict())

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_acc_state_dict = ae.state_dict()

        if re_loss < best_loss:
            print("new best loss", re_loss)
            best_state_dict = ae.state_dict()
            best_loss = re_loss
            best_epoch = j

    return best_accuracy, best_epoch, best_state_dict, results, ae.state_dict(), best_acc_state_dict, models


def train_beta_vcca_single(ae, z_dim, beta, train_loader, validation_loader, test_loader, num_epochs, device, suppress_output=False):
    optimizer_AE = torch.optim.Adam(ae.parameters(), lr=1e-3)

    best_accuracy = 0.0
    best_epoch = 0
    best_loss = np.inf

    results = TrainingResults()

    best_state_dict = ae.state_dict()
    best_acc_state_dict = ae.state_dict()
    models = []

    for j in range(num_epochs):
        running_re_loss = 0.0
        
        ae.train()
        for i, data in enumerate(train_loader, 0): 
            x, y = data[0].to(device), data[1].to(device)

            optimizer_AE.zero_grad()

            x_hat, y_hat, z_mu, z_logvar, z = ae(x,y)
            
            loss = vcca_single_loss(x_hat, x, y_hat, y, z_mu, z_logvar, beta)
            loss.backward()
            optimizer_AE.step()

            running_re_loss += loss.item()

        ae.eval()
        re_loss = running_re_loss/(i+1)
        
        results.training_re_losses.append(re_loss)
        
        if not suppress_output:
            print(j, "TRAINING loss: ", re_loss)

        # Train classifier
        x, y, rot_x, rot_y, labels = next(iter(validation_loader))
        x_hat, y_hat, z_mu, z_logvar, z = ae(x.to(device), y.to(device))
        train_z, train_labels, train_rot_x, train_rot_y = z_mu.detach().cpu().numpy(), labels.cpu().numpy(), rot_x.cpu().numpy(), rot_y.cpu().numpy()
        clf, regr_a, regr_b = [], [], []
        inputs = [train_z]
        for i in range(len(inputs)):
            clf.append(LinearSVC(random_state=0, tol=1e-5))
            regr_a.append(LinearSVR(random_state=0, tol=1e-5))
            regr_b.append(LinearSVR(random_state=0, tol=1e-5))
        for i in range(len(inputs)):
            clf[i].fit(inputs[i], train_labels)
            regr_a[i].fit(inputs[i], train_rot_x)
            regr_b[i].fit(inputs[i], train_rot_y)
        #evaluate
        x, y, rot_x, rot_y, labels = next(iter(test_loader))
        x_hat, y_hat, z_mu, z_logvar, z = ae(x.to(device),y.to(device))
        test_z, test_labels, test_rot_x, test_rot_y = z_mu.detach().cpu().numpy(), labels.cpu().numpy(), rot_x.cpu().numpy(), rot_y.cpu().numpy()
        inputs = [test_z]

        class_scores, rot_x_scores, rot_y_scores = [], [], []
        for i in range(len(inputs)):
            class_scores.append(clf[i].score(inputs[i], test_labels))
            rot_x_scores.append(regr_a[i].score(inputs[i], test_rot_x))
            rot_y_scores.append(regr_b[i].score(inputs[i], test_rot_y))
        accuracy = .3333 * (class_scores[0] + rot_x_scores[0] + rot_y_scores[0])
        if not suppress_output:
            print('accuracy: ', accuracy)

        results.class_predictions.append(class_scores)
        results.rotx_predictions.append(rot_x_scores)
        results.roty_predictions.append(rot_y_scores)

        # Saving Loss curves
        if j % 10 == 0:
            models.append(ae.state_dict())

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_acc_state_dict = ae.state_dict()

        if re_loss < best_loss:
            if not suppress_output:
                print("new best loss", re_loss)
            else:
                print("new best loss", re_loss, " epoch", j)
            best_state_dict = ae.state_dict()
            best_loss = re_loss
            best_epoch = j

    return best_accuracy, best_epoch, best_state_dict, results, ae.state_dict(), best_acc_state_dict, models


def train_abc_vcca_private(ae, z_dim, beta, hx_dim, alpha, hy_dim, gamma, train_loader, validation_loader, test_loader, num_epochs, device, suppress_output=False):
    """
    alpha (a) weights hx KLD
    beta (b) weights z KLD
    gamma (c) weights hy KLD
    """
    optimizer_AE = torch.optim.Adam(ae.parameters(), lr=1e-3)

    best_accuracy = 0.0
    best_epoch = 0
    best_loss = np.inf

    results = TrainingResults()

    best_state_dict = ae.state_dict()
    best_acc_state_dict = ae.state_dict()
    models = []

    for j in range(num_epochs):
        running_re_loss = 0.0
        
        ae.train()
        for i, data in enumerate(train_loader, 0): 
            x, y = data[0].to(device), data[1].to(device)

            optimizer_AE.zero_grad()
            
            x_hat, y_hat, z_mu, z_logvar, hx_mu, hx_logvar, hy_mu, hy_logvar, z, hx, hy = ae(x,y)
            
            loss = vcca_private_loss(x_hat, x, y_hat, y, z_mu, z_logvar, hx_mu, hx_logvar, hy_mu, hy_logvar, alpha, beta, gamma)
            loss.backward()
            optimizer_AE.step()

            running_re_loss += loss.item()

        ae.eval()
        re_loss = running_re_loss/(i+1)
        
        results.training_re_losses.append(re_loss)
        
        if not suppress_output:
            print(j, "TRAINING loss: ", re_loss)

        # Train classifier
        x, y, rot_x, rot_y, labels = next(iter(validation_loader))
        x_hat, y_hat, z_mu, z_logvar, hx_mu, hx_logvar, hy_mu, hy_logvar, z, hx, hy = ae(x.to(device),y.to(device))
        train_z, train_hx, train_hy, train_labels, train_rot_x, train_rot_y = z_mu.detach().cpu().numpy(), hx_mu.detach().cpu().numpy(), hy_mu.detach().cpu().numpy(), labels.cpu().numpy(), rot_x.cpu().numpy(), rot_y.cpu().numpy()
        clf, regr_a, regr_b = [], [], []
        inputs = [train_z, train_hx, train_hy]
        for i in range(3):
            clf.append(LinearSVC(random_state=0, tol=1e-5))
            regr_a.append(LinearSVR(random_state=0, tol=1e-5))
            regr_b.append(LinearSVR(random_state=0, tol=1e-5))
        for i in range(3):
            clf[i].fit(inputs[i], train_labels)
            regr_a[i].fit(inputs[i], train_rot_x)
            regr_b[i].fit(inputs[i], train_rot_y)
        #evaluate
        x, y, rot_x, rot_y, labels = next(iter(test_loader))
        x_hat, y_hat, z_mu, z_logvar, hx_mu, hx_logvar, hy_mu, hy_logvar, z, hx, hy = ae(x.to(device),y.to(device))
        test_z, test_hx, test_hy, test_labels, test_rot_x, test_rot_y = z_mu.detach().cpu().numpy(), hx_mu.detach().cpu().numpy(), hy_mu.detach().cpu().numpy(), labels.cpu().numpy(), rot_x.cpu().numpy(), rot_y.cpu().numpy()
        inputs = [test_z, test_hx, test_hy]

        class_scores, rot_x_scores, rot_y_scores = [], [], []
        for i in range(3):
            class_scores.append(clf[i].score(inputs[i], test_labels))
            rot_x_scores.append(regr_a[i].score(inputs[i], test_rot_x))
            rot_y_scores.append(regr_b[i].score(inputs[i], test_rot_y))
        accuracy = .3333 * (class_scores[0] + rot_x_scores[1] + rot_y_scores[2])
        if not suppress_output:
            print('accuracy: ', accuracy)

        results.class_predictions.append(class_scores)
        results.rotx_predictions.append(rot_x_scores)
        results.roty_predictions.append(rot_y_scores)

        # Saving loss curves
        if j % 10 == 0:
            models.append(ae.state_dict())

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_acc_state_dict = ae.state_dict()

        if re_loss < best_loss:
            if not suppress_output:
                print("new best loss", re_loss)
            else:
                print("new best loss", re_loss, " epoch", j)
            best_state_dict = ae.state_dict()
            best_loss = re_loss
            best_epoch = j

    return best_accuracy, best_epoch, best_state_dict, results, ae.state_dict(), best_acc_state_dict, models

