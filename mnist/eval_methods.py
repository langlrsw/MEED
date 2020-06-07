# -*- coding: utf-8 -*-

try:
    import cPickle as pickle
except:
    import pickle
import os
import numpy as np
import keras

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

import tensorflow as tf
 

from matplotlib import pyplot as plt

def img2Dto3D(x):
    
    return np.repeat(x,3,axis=2)

def add_red_line(x):
    
    x[:,-1,0] = 1.
    x[:,-1,1] = 0.
    x[:,-1,2] = 0.
    
    return x
    

def show_example(x_test,selection,j):
    
    x1 = img2Dto3D(x_test[j])
    x2 = img2Dto3D(selection[j])
    x3 = img2Dto3D(x_test[j]*selection[j])
    x4 = img2Dto3D(x_test[j]*(1.-selection[j]))
    
    method_name = 'Ours'
    mkdir('images_new')
    plt.imsave('images_new/%s_order_%d_x1.png' %(method_name,j),x1) 
    plt.imsave('images_new/%s_order_%d_x2.png' %(method_name,j),x2) 
    plt.imsave('images_new/%s_order_%d_x3.png' %(method_name,j),x3) 
    plt.imsave('images_new/%s_order_%d_x4.png' %(method_name,j),x4) 
    
    x1 = add_red_line(x1)
    x2 = add_red_line(x2)
    x3 = add_red_line(x3)
    
    mat = np.concatenate([x1,
                          x2,
                          x3,
                          x4],
                         axis=1)
    plt.imshow(mat)
    plt.show()
            
    plt.imsave('images_new/%s_order_%d_mat.png' %(method_name,j),mat) 
            
    return

def eval_fidelity(model_As, selection, x_test, y_test,flag_only_mask=False,flag_show_example=False,id_to_word=None,mx=None):
    if flag_only_mask:
        selected_x = selection
    else:
        if mx is None:
            selected_x = selection * x_test 
        else:
            selected_x = selection * (x_test-mx) + mx
    y_pred = model_As.predict(selected_x,batch_size = 2048)
    
#     idx_pos = np.where(np.argmax(y_test, axis=-1) == np.argmax(y_pred, axis=-1))[0]
    idx_pos = np.argsort(np.max(y_pred,axis=1))
    idx_neg = np.where(np.argmax(y_test, axis=-1) != np.argmax(y_pred, axis=-1))[0]
 
#     print('idx_neg',idx_neg)
    
    if flag_show_example:
#         np.random.shuffle(idx_pos)
#         idx_pos = np.array([ 3580, 15370, 23023, 13813,   389,   502,  9764,  1248, 17205,
#        19509])
#         idx_pos = np.array([1231, 1106, 1718,  134,  996, 1322,  209,   53, 1952,  658])
        idx_pos = np.array([1004,  848,  185, 1326, 1432, 1428,  842,  130, 1634, 1609,  482,
        592, 1776,  613, 1922,  175,  216, 1221, 1307,  924])
        print('idx_pos[:20]',idx_pos[:20])
        for i in range(20):
            j = idx_pos[i]
            print(y_pred[j])
            print('fidelity, consistent, order=%d, y_test = %d, y_pred = %d' % (j,np.argmax(y_test[j]),np.argmax(y_pred[j])))
            
#             mat = np.concatenate([x_test[j],selection[j],x_test[j]*selection[j]],axis=1)
#             plt.imshow(mat[:,:,0])
#             plt.show()
            show_example(x_test,selection,j)
             

        np.random.shuffle(idx_neg)
#         idx_neg = np.array([1312, 1185,  645, 1245,  842, 1557,   73,  816, 1304, 1132])
#         idx_neg = np.array([ 5629, 17540,  8711, 23822, 17890,  6884, 13056,  7703,   910,
#        10162])
        idx_neg = np.array(  [1179,  848,  816, 1015, 1304,  613,   73,  784, 1312,   97])
        print('idx_neg[:10]',idx_neg[:10])
        for i in range(10):
            j = idx_neg[i]
            print('fidelity, inconsistent, order=%d, y_test = %d, y_pred = %d' % (j,np.argmax(y_test[j]),np.argmax(y_pred[j])))
#             mat = np.concatenate([x_test[j],selection[j],x_test[j]*selection[j]],axis=1)
#             plt.imshow(mat[:,:,0])
#             plt.show()
            show_example(x_test,selection,j)

    test_acc = np.mean(np.argmax(y_test, axis=-1) == np.argmax(y_pred, axis=-1))

    return test_acc


def eval_infidelity(model_Au, selection, x_test, y_test,flag_only_mask=False,mx=None):
    selection = np.ones_like(selection) - selection
    
    if flag_only_mask:
        selected_x = selection
    else:
        if mx is None:
            selected_x = selection * x_test 
        else:
            selected_x = selection * (x_test-mx) + mx
    y_pred = model_Au.predict(selected_x,batch_size = 2048)
     
    test_acc = np.mean(np.argmax(y_test, axis=-1) == np.argmax(y_pred, axis=-1))
    return test_acc

def soft2hard_explanation(scores,k):
    n,h,w,_ = scores.shape
    scores = np.reshape(scores,(n,h*w))
    new_scores = np.zeros(scores.shape)
    
    selected = np.argsort(scores,axis=1)[:,-k:] 
    for i in range(scores.shape[0]):
        new_scores[i][selected[i]] = 1.0
        
    new_scores = np.reshape(new_scores,(n,h,w,1))
    
    return new_scores

def norm_x(x):
    x_shape = x.shape
    n = x*x
    for _ in range(len(x_shape)-1):
        n = np.sum(n,axis=-1)
    return np.sqrt(n)

def eval_sensitivity2(x, Explainer, epsilon, sen_N, y=None,selection=None):
    
    if selection is None:
        if y is not None:
            selection = Explainer.predict([x,y],batch_size = 32)
        else:
            selection = Explainer.predict(x,batch_size = 32)
            
    max_sens = []
    noise_size = [sen_N]
    noise_size.extend(x.shape[1:])
    
    norm = norm_x(selection)
    math_diff = - np.ones(x.shape[0])
    
    for _ in range(sen_N):
        math_diff = -1.
        noise_samples = np.random.uniform(-epsilon, epsilon, x.shape)
        x_noisy = x + noise_samples
        if y is not None:
            selection_noisy = Explainer.predict([x_noisy, y],batch_size=32)
        else:
            selection_noisy = Explainer.predict(x_noisy,batch_size=32)
        max_sens_single = norm_x(selection - selection_noisy)
        math_diff = np.maximum(math_diff,max_sens_single)
 
    max_sen = np.mean(math_diff/norm)
    return max_sen


def to_categorical(y,nb_classes):
    return np.eye(nb_classes)[y]

def predict_selection_tf(sess,pred_val,x_val,grad_sum,x_placeholder,y_placeholder):
    
    num_classes = pred_val.shape[1]
    
    pred_test_one_hot = to_categorical(np.argmax(pred_val,axis=1),num_classes)
        
    idx = np.arange(x_val.shape[0])
    num_batch = int(np.ceil(x_val.shape[0]/float(256)))
    selection_test = []
    for i_batch in range(num_batch):
        if i_batch == num_batch - 1:
            idx_i = idx[i_batch*256:]
        else:
            idx_i = idx[i_batch*256:(i_batch+1)*256]

        selection_test_i = sess.run(grad_sum, feed_dict={x_placeholder: x_val[idx_i], y_placeholder: pred_test_one_hot[idx_i]})
        selection_test.append(selection_test_i)
    selection_test = np.vstack(selection_test)
    
    return selection_test




def eval_sensitivity3(x, Explainer, epsilon, sen_N,sess,grad_sum,x_placeholder,y_placeholder,M, y=None,selection=None):
    
    if selection is None:
        if y is not None:
            selection = Explainer.predict([x,y],batch_size = 32)
        else:
            selection = Explainer.predict(x,batch_size = 32)
            
    max_sens = []
    noise_size = [sen_N]
    noise_size.extend(x.shape[1:])
    
    norm = norm_x(selection)
    math_diff = - np.ones(x.shape[0])
    
    for _ in range(sen_N):
        math_diff = -1.
        noise_samples = np.random.uniform(-epsilon, epsilon, x.shape)
        x_noisy = x + noise_samples
#         y = M.predict(x_noisy,batch_size = 32)
        selection_noisy = predict_selection_tf(sess,y,x_noisy,grad_sum,x_placeholder,y_placeholder)
        max_sens_single = norm_x(selection - selection_noisy)
        math_diff = np.maximum(math_diff,max_sens_single)
 
    max_sen = np.mean(math_diff/norm)
    return max_sen

def eval_without_approximator(model_explainer, model_predictor, x_test, y_test,flag_with_y=False,k=10,flag_only_mask=False,flag_show_example=False,id_to_word=None,mx=None,selection=None):
 
    if selection is None:
    
        if flag_with_y:
            selection = model_explainer.predict([x_test,y_test],batch_size = 2048)
        else:
            selection = model_explainer.predict(x_test,batch_size = 2048)
        
    selection = np.reshape(selection,x_test.shape)
    selection = soft2hard_explanation(selection,k)
    acc_fidelity = eval_fidelity(model_predictor, selection, x_test, y_test,flag_only_mask=flag_only_mask,flag_show_example=flag_show_example,id_to_word=id_to_word,mx=mx)
    acc_infidelity = eval_infidelity(model_predictor, selection, x_test, y_test,flag_only_mask=flag_only_mask,mx=mx)

    return acc_fidelity, acc_infidelity



def eval_with_approximator(model_explainer, model_As,model_Au, x_train, y_train, x_test, y_test, epochs=5,flag_train=True,flag_with_y=False,k=10,flag_only_mask=False,flag_eval_train=False,flag_train_test=False,
                          selection_train=None,selection_test=None):
 
    if selection_train is None:
        if flag_with_y:
            selection_train = model_explainer.predict([x_train,y_train],batch_size=2048) 
            selection_test = model_explainer.predict([x_test,y_test],batch_size=2048) 
        else:
            selection_train = model_explainer.predict(x_train,batch_size=2048) 
            selection_test = model_explainer.predict(x_test,batch_size=2048) 
        
    selection_train = np.reshape(selection_train,x_train.shape)
    selection_test = np.reshape(selection_test,x_test.shape)
    
    selection_train = soft2hard_explanation(selection_train,k)
    selection_test = soft2hard_explanation(selection_test,k)
 
    if flag_train:
        
#         print('fidelity: np.min(np.sum(x_train_f==0,axis=1))',np.min(np.sum(x_train_f==0,axis=1)))
        if flag_train_test:
            model_As.fit(x_test*selection_test, y_test, epochs=epochs,verbose=0)
        else:
            model_As.fit(x_train*selection_train, y_train, epochs=epochs,verbose=0)

    if flag_eval_train:
        acc_fidelity = eval_fidelity(model_As, selection_train, x_train, y_train,flag_only_mask=flag_only_mask)
    else:
        acc_fidelity = eval_fidelity(model_As, selection_test, x_test, y_test,flag_only_mask=flag_only_mask)
 
    if flag_train:
        
        
#         print('infidelity: np.min(np.sum(x_train_if==0,axis=1))',np.min(np.sum(x_train_if==0,axis=1)))
        if flag_train_test:
            model_Au.fit(x_test*(1.-selection_test), y_test, epochs=epochs,verbose=0)
        else:
            model_Au.fit(x_train*(1 - selection_train), y_train, epochs=epochs,verbose=0)

    if flag_eval_train:
        acc_infidelity = eval_infidelity(model_Au, selection_train, x_train, y_train,flag_only_mask=flag_only_mask)
    else:
        acc_infidelity = eval_infidelity(model_Au, selection_test, x_test, y_test,flag_only_mask=flag_only_mask)

    return acc_fidelity, acc_infidelity,model_As,model_Au


def eval(model_explainer, ori_predictor, get_approximator, x_train, y_train, x_test, y_test, epochs=10):

    f_original, if_original = eval_without_approximator(model_explainer, ori_predictor, x_test, y_test)

    f_approximator, if_approximator = eval_with_approximator(model_explainer, get_approximator, x_train, y_train, x_test, y_test, epochs)
    return f_original, if_original, f_approximator, if_approximator


def get_model():
    input = Input([191])
    out = Dense(2, activation='softmax')(input)
    return Model(input, out)

if __name__ == '__main__':
    import keras
    from keras.models import Model
    from keras.layers import Dense, Input
    from keras.utils import to_categorical

    n = 10000
    selection = np.random.randn(n, 191)
    selection_ones = np.ones_like(selection)

    x_train = np.random.randn(n, 191)
    x_test = np.random.randn(n, 191)

    y = to_categorical(np.random.randint(0, 2, size=(n,)), 2)
    y_selector = to_categorical(np.random.randint(0, 2, size=(n,)), 191)

    input = Input([191])
    out = Dense(2, activation='softmax')(input)

    predictor = Model(input, out)
    out_2 = Dense(2, activation='softmax')(input)
    predictor_2 = Model(input, out_2)


    predictor.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
    predictor_2.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
    predictor.fit(x_train, y, epochs=1)



    out_2 = Dense(191, activation='softmax')(input)
    selector = Model(input, out_2)
    selector.compile(loss=keras.losses.binary_crossentropy,
                     optimizer=keras.optimizers.Adadelta(),
                     metrics=['accuracy'])
    selector.fit(x_train, y_selector, epochs=1)



    y_pred_train = predictor.predict(x_train)
    y_pred_test = predictor.predict(x_test)
 

    f_original, if_original, f_approximator, if_approximator = eval(selector, predictor, get_model, x_train, y_pred_train, x_test, y_pred_test, 1)

    print(f_original)
    print(if_original)
    print(f_approximator)
    print(if_approximator)
