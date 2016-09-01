# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 17:24:54 2016

@author: rbodo
"""

i = 2
input_model = model_lib.load_ann(settings['path_wd'], settings['filename_ann'])
model = input_model['model']
model.evaluate(X_test[i:i+1], Y_test[i:i+1])
plt.hist([np.argmax(Y_test[i:i+1]) for i in range(100)], bins=100)
plt.hist([np.argmax(get_activ(X_test[i:i+1])) for i in range(100)])
plt.imshow(np.transpose(X_test[i], (1, 2, 0)))
np.argmax(Y_test[i:i+1])
get_activ = get_activ_fn_for_layer(model, -1)
np.argmax(get_activ(X_test[i:i+1]))
score = model_lib.evaluate(input_model['val_fn'], X_test, Y_test)