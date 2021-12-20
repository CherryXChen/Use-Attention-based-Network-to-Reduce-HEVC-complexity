import tensorflow as tf
import config as cf

#-----------------------------------------------------------------------------
#BATCH_SIZE = 64
#IMAGE_SIZE = 64
#NUM_CHANNELS = 1
#NUM_EXT_FEATURES_CNN = 1
#NUM_LABEL_BYTES = 16
#GOP_LENGTH = 4
#VECTOR_LENGTH_LIST = [64, 128, 256]
#VECTOR_LENGTH = sum(VECTOR_LENGTH_LIST)
#LSTM_DEPTH = 1
#LSTM_MAX_LENGTH = 20
#LSTM_OVERLAP_STRIDE = 10
#LSTM_READ_LENGTH = LSTM_MAX_LENGTH
#LSTM_OUTPUT_LENGTH = LSTM_MAX_LENGTH
#NUM_EXT_FEATURES = LSTM_MAX_LENGTH * 2 + 1
#efs = [lstm_length, qps, i_frame_in_GOP] 1,20,20
#one hot gop: [num, 20, 4]
#-----------------------------------------------------------------------------

BATCH_SIZE = cf.BATCH_SIZE
IMAGE_SIZE = cf.IMAGE_SIZE
NUM_CHANNELS = cf.NUM_CHANNELS 

NUM_EXT_FEATURES = cf.NUM_EXT_FEATURES
NUM_LABEL_BYTES = cf.NUM_LABEL_BYTES
GOP_LENGTH = cf.GOP_LENGTH
VECTOR_LENGTH = cf.VECTOR_LENGTH
VECTOR_LENGTH_LIST = cf.VECTOR_LENGTH_LIST
LSTM_MAX_LENGTH = cf.LSTM_MAX_LENGTH
LSTM_READ_LENGTH = cf.LSTM_READ_LENGTH
LSTM_OUTPUT_LENGTH = cf.LSTM_OUTPUT_LENGTH
LSTM_DEPTH = cf.LSTM_DEPTH

NUM_VECTOR_SIZE_64 = VECTOR_LENGTH_LIST[0]
NUM_VECTOR_SIZE_32 = VECTOR_LENGTH_LIST[1]
NUM_VECTOR_SIZE_16 = VECTOR_LENGTH_LIST[2]

NUM_HIDDEN_SIZE_64 = NUM_VECTOR_SIZE_64 
NUM_HIDDEN_SIZE_32 = NUM_VECTOR_SIZE_32 
NUM_HIDDEN_SIZE_16 = NUM_VECTOR_SIZE_16 
      
NUM_DENLAYER2_FEATURES_64 = 48
NUM_DENLAYER2_FEATURES_32 = 96       
NUM_DENLAYER2_FEATURES_16 = 192  

MAX_GRAD_NORM = 5
NUM_EXT_FEATURES = 1 + GOP_LENGTH # QP and i_frame_in_GOP (one hot)

# weight initialization
def weight_variable(shape, name=None, is_reuse_var=False):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable("full_connect_w", shape) # KEY POINT. To share variables, invoke "get_variable()" rather than "Variable()"

def bias_variable(shape, name=None, is_reuse_var=False):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable("full_connect_b", shape) # KEY POINT. To share variables, invoke "get_variable()" rather than "Variable()"

def add_weight_decay_to_losses(var,wd):
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)	

# convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
# pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def aver_pool(x, k_width):
    return tf.nn.avg_pool(x, ksize=[1, k_width, k_width, 1], strides=[1, k_width, k_width, 1], padding='SAME')

def activate(x, acti_mode, scope=None):
    if acti_mode==0:
        return x
    elif acti_mode==1:
        return tf.nn.relu(x)
    elif acti_mode==2:
        return tf.nn.sigmoid(x)
    elif acti_mode==3:
        return (tf.nn.tanh(x) + x) / 2
    elif acti_mode==4:
        return (tf.nn.sigmoid(x) + x) / 2    
    elif acti_mode==5:
        return tf.nn.leaky_relu(x)   

def expand_dim(x, axis, num_dup):
    x_list=[]
    for i in range(num_dup):
        x_list.append(x)
    return tf.stack(x_list, axis)

def full_connect(x, num_filters_in, num_filters_out, acti_mode, keep_prob=1, name_w=None, name_b=None, is_reuse_var=False): 
    w_fc = weight_variable([num_filters_in, num_filters_out], name_w, is_reuse_var)
    b_fc = bias_variable([num_filters_out], name_b, is_reuse_var)
    x_t = tf.reshape(x, (-1, num_filters_in))
    #print('x_t: ',x_t)
    h_fc = tf.matmul(x_t, w_fc) + b_fc
    h_fc = tf.reshape(h_fc, (-1, LSTM_READ_LENGTH, num_filters_out))
    h_fc = activate(h_fc, acti_mode)
    h_fc = tf.cond(keep_prob < tf.constant(1.0), lambda: tf.nn.dropout(h_fc, keep_prob), lambda: h_fc)
    #print(h_fc) 
    return h_fc

'''
Following modify by Tianrui Chen
'''
#Layer Normalization module
def ln(x, params_shape, epsilon = 1e-8, scope="ln"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # input_shape = x.get_input()
        # params_shape = input_shape[-1:]
        #get mean and variance of each layer
        mean, var = tf.nn.moments(x, [-1], keep_dims=True)
        beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer)
        gamma = tf.get_variable('gamma', params_shape, initializer=tf.zeros_initializer)
        norm = (x - mean) / ((var + epsilon) ** 0.5)
        outputs = gamma * norm + beta
        
    return outputs

def scaled_dot_product(Q, K, V,
                       dropout_rate=0.1,
                       training=True,
                       scope="scaled_dot_product"):
    '''
    Q: [N, T_q, d_k].
    K: [N, T_k, d_k].
    V: [N, T_k, d_v].
    dropout_rate
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]
        
        output = tf.matmul(Q, tf.transpose(K, [0,2,1])) #[N, T_q, T_k]
        output = output / d_k ** 0.5
        #output = tf.nn.softmax(output)
        output = tf.nn.sigmoid(output)
        #output = tf.transpose(output, [0,2,1]) #[N, T_k,T_q]
        output = tf.matmul(output, V) #[N, ]
    
    return output

def multihead_attention(queries, keys, values, d_model,
                        num_heads=1, 
                        dropout_rate=0,
                        training=True,
                        scope="multihead_attention"):
    '''
    queries: [N, T_q, d_model].
    keys: [N, T_k, d_model].
    values: [N, T_k, d_model].
    num_heads: Number of heads.
        
    Returns: (N, T_q, C)
    '''
    d_model = d_model
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        #only one kernel is applied
        Q = tf.layers.dense(queries, d_model, use_bias=True)
        K = tf.layers.dense(keys, d_model, use_bias=True)
        V = tf.layers.dense(values, d_model, use_bias=True)
        
        #Multi-head
        Q_M = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0)
        K_M = tf.concat(tf.split(K, num_heads, axis=-1), axis=0)
        V_M = tf.concat(tf.split(V, num_heads, axis=-1), axis=0)
        
        output = scaled_dot_product(Q_M, K_M, V_M)
        output = tf.concat(tf.split(output, num_heads, axis=0), axis=2)
        
        output = ln(output, d_model)
        return output

def resi_connention(X, Y):
    #concatenate the input and output from some layers
    return X + Y

def attention(x, num_x_size_x, num_input_hidden_size, isdrop,
              num_denlayers_2_features, num_denlayers_3_features,
              qp_list, i_frame_in_GOP_one_hot_list, acti_mode_fc,
              scope = 'Attention', name_string ='64', num_heads=1, batch_size = BATCH_SIZE):
    
    x_re = tf.reshape(x, [-1, LSTM_READ_LENGTH, num_x_size_x])
    # x_re: [20, batch_size, 1, 64/128/256]
    atten_ini = tf.split(x_re, LSTM_READ_LENGTH, axis=1)
    atten_inputs = []
    for time_step in range(LSTM_READ_LENGTH):
        atten_ini_one = tf.reshape(atten_ini[LSTM_READ_LENGTH - 1 - time_step], (-1, num_x_size_x))
        atten_inputs.append(tf.concat([atten_ini_one, i_frame_in_GOP_one_hot_list[time_step]], axis=1))
    #print('atten_inputs: ', atten_inputs)

    # GOP_temp = tf.split(i_frame_in_GOP_one_hot_list, batch_size, axis=1)
    # qp_temp = tf.split(qp_list, batch_size, axis=1)
    # atten_inputs = []
    # efs = []
    # for batch in range(batch_size):
    #     GOP_temp_one = tf.reshape(GOP_temp[batch], [-1, GOP_LENGTH])
    #     qp_temp_one = tf.reshape(qp_temp[batch], [-1, 1])
    #     efs.append(tf.concat([qp_temp_one, GOP_temp_one], axis=1))
    #     atten_inputs.append(tf.concat([x_re[batch], GOP_temp_one], axis=1))
    
    atten_inputs = tf.transpose(atten_inputs, [1, 0, 2])
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        atten_outputs = multihead_attention(atten_inputs, atten_inputs, atten_inputs, num_x_size_x + 4)
        atten_outputs += atten_inputs

        #print('atten_outputs: ', atten_outputs)

        atten_outputs_re = tf.reshape(atten_outputs, [-1, LSTM_READ_LENGTH, num_x_size_x + 4])
        fc1_ini = tf.split(atten_outputs_re, LSTM_READ_LENGTH, axis=1)
        fc1_inputs = []

        for time_step in range(LSTM_READ_LENGTH):
            efs = tf.concat([qp_list[time_step], i_frame_in_GOP_one_hot_list[time_step]], axis = 1)
            fc1_ini_one = tf.reshape(fc1_ini[time_step], (-1, num_x_size_x + 4))
            fc1_inputs.append(tf.concat([fc1_ini_one, efs], axis=1))

        fc1_inputs = tf.transpose(fc1_inputs, [1,0,2])
        
        with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
            fc1_outputs = full_connect(fc1_inputs, num_input_hidden_size + 2 * GOP_LENGTH + 1, num_denlayers_2_features,
                                    acti_mode = acti_mode_fc, keep_prob = 1-isdrop*0.2, is_reuse_var=True)
        
        fc1_outputs_re = tf.reshape(fc1_outputs, [-1, LSTM_READ_LENGTH, num_denlayers_2_features])
        fc2_ini = tf.split(fc1_outputs_re, LSTM_READ_LENGTH, axis=1)
        fc2_inputs = []

        for time_step in range(LSTM_READ_LENGTH):
            efs = tf.concat([qp_list[time_step], i_frame_in_GOP_one_hot_list[time_step]], axis = 1)
            fc2_ini_one = tf.reshape(fc2_ini[time_step], (-1, num_denlayers_2_features))
            fc2_inputs.append(tf.concat([fc2_ini_one, efs], axis=1))

        fc2_inputs = tf.transpose(fc2_inputs, [1,0,2])
        
        with tf.variable_scope('fc2', reuse=tf.AUTO_REUSE):
            fc2_outputs = full_connect(fc2_inputs, num_denlayers_2_features + GOP_LENGTH + 1, num_denlayers_3_features,
                                    acti_mode = 2, keep_prob = 1, is_reuse_var=True)
    #fc2_outputs.reverse()
    y_pred_flat_list = tf.reverse(fc2_outputs, [1])
    #y_pred_flat_list = tf.transpose(y_pred_flat_list, [1, 0, 2])
    h_fc3_line_list = []
    #print('y_pred_flat_list: ', y_pred_flat_list)
    return y_pred_flat_list, h_fc3_line_list

def lstm(x, num_x_size_x, num_input_hidden_size, isdrop, 
        num_denlayers_2_features, num_denlayers_3_features, 
         qp_list, i_frame_in_GOP_one_hot_list, acti_mode_fc,
         variable_scope = 'RNN', name_string ='64'):
    def lstm_cell():
        return tf.contrib.rnn.LSTMCell(num_input_hidden_size, forget_bias = 1.0, cell_clip = 5.0, state_is_tuple=True)
    
    attn_cell = lstm_cell
    
    def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob = 1 - isdrop * 0.5)    
    
    cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(LSTM_DEPTH)], state_is_tuple=True)
    
    # tf.shape(x)[0] is batch size
    initial_state = cell.zero_state(tf.shape(x)[0], tf.float32)
    # cell_inputs: [batch_size, 20, 64/128/256]
    cell_inputs = tf.reshape(x, [-1, LSTM_READ_LENGTH, num_x_size_x])
    #print(cell_inputs)
    # cell_inputs: [20, batch_size, 1, 64/128/256]
    cell_inputs = tf.split(cell_inputs, LSTM_READ_LENGTH, axis=1)
    for time_step in range(LSTM_READ_LENGTH):
        cell_inputs[time_step] = tf.reshape(cell_inputs[time_step], [-1, num_x_size_x])
    # cell_inputs: [20, batch_size, 64/128/256]
    cell_inputs.reverse() 

    cell_outputs = []
    y_pred_flat_list = []
    h_fc3_line_list = []    
    state = initial_state
    with tf.variable_scope(variable_scope):
        for time_step in range(LSTM_READ_LENGTH):
            if time_step > 0: 
                tf.get_variable_scope().reuse_variables()
            (cell_output_one_step, state) = cell(cell_inputs[time_step], state)
            cell_outputs.append(cell_output_one_step)    
            if time_step < LSTM_OUTPUT_LENGTH:
                #print(qp_list[time_step])
                #print(i_frame_in_GOP_one_hot_list[time_step])
                #print(cell_outputs[time_step])
                efs = tf.concat([qp_list[time_step], i_frame_in_GOP_one_hot_list[time_step]], axis = 1)
                with tf.variable_scope('fc2'):
                    h_fc1 = tf.concat([cell_outputs[time_step], efs], axis = 1)
                    h_fc2 = full_connect(h_fc1, num_input_hidden_size + GOP_LENGTH + 1, num_denlayers_2_features,
                                            acti_mode = acti_mode_fc, keep_prob = 1-isdrop*0.2, is_reuse_var=(time_step>0)) 
                with tf.variable_scope('fc3'):
                    h_fc2 = tf.concat([h_fc2, efs], axis = 1)
                    y_pred_flat = full_connect(h_fc2, num_denlayers_2_features + GOP_LENGTH + 1, num_denlayers_3_features,
                                             acti_mode = 2, keep_prob = 1, is_reuse_var=(time_step>0))

                y_pred_flat_list.append(y_pred_flat)
    
    y_pred_flat_list.reverse()
    h_fc3_line_list.reverse()

    return y_pred_flat_list, h_fc3_line_list

def net(x,y_,ef,isdrop,global_step,learning_rate_init,momentum,decay_step, decay_rate, limit_grad = False, is_balance = False):
    '''
    ef: extra futures: [lstm_length, qps, i_frame_in_GOP] 1,20,20
    lstm_length: [batch_size, 1]
    qp: [batch_size, 20]
    i_frame_in_GOP: [batch, 20]
    i_frame_in_GOP_one_hot: [batch, 20, 4]
    '''
    lstm_length = tf.slice(ef, [0,0],[-1,1])
    qp = tf.slice(ef, [0,1],[-1,LSTM_MAX_LENGTH])
    i_frame_in_GOP = tf.slice(ef, [0,LSTM_MAX_LENGTH+1],[-1,LSTM_MAX_LENGTH])
    i_frame_in_GOP_one_hot = tf.reshape(tf.one_hot(tf.to_int32(i_frame_in_GOP), depth = GOP_LENGTH, axis = 2), [-1, LSTM_MAX_LENGTH, GOP_LENGTH])
    
    #Normalize qp
    qp = qp / 51.0
    
    y_flat_16_list = []
    y_flat_32_list = []
    y_flat_64_list = []
    y_flat_valid_32_list = []
    y_flat_valid_16_list = []
    
    qp_list = []
    i_frame_in_GOP_one_hot_list = []
    
    '''
    LSTM_OUTPUT_LENGTH: 20
    qp_list: [20, batch_size, 1]
    i_frame_in_GOP_one_hot_list: [20, batch_size, 4]
    '''
    
    for i in range(LSTM_OUTPUT_LENGTH):
        y_image = tf.reshape(tf.slice(y_, [0, i, 0], [-1,1,-1]), [-1, 4, 4, 1])
        
        y_image_16 = tf.nn.relu(y_image-2)
        y_image_32 = tf.nn.relu(aver_pool(y_image, 2)-1)-tf.nn.relu(aver_pool(y_image, 2)-2)
        y_image_64 = tf.nn.relu(aver_pool(y_image, 4)-0)-tf.nn.relu(aver_pool(y_image, 4)-1)
        y_image_valid_32 = tf.nn.relu(aver_pool(y_image, 2)-0)-tf.nn.relu(aver_pool(y_image, 2)-1)
        y_image_valid_16 = tf.nn.relu(y_image-1)-tf.nn.relu(y_image-2)
        
        y_flat_16_list.append(tf.reshape(y_image_16, [-1, 16]))
        y_flat_32_list.append(tf.reshape(y_image_32, [-1, 4]))
        y_flat_64_list.append(tf.reshape(y_image_64, [-1, 1]))
        
        y_flat_valid_32_list.append(tf.reshape(y_image_valid_32, [-1, 4]))
        y_flat_valid_16_list.append(tf.reshape(y_image_valid_16, [-1, 16]))
        
        qp_list.append(tf.reshape(tf.slice(qp, [0, i], [-1, 1]), [-1, 1]))
        i_frame_in_GOP_one_hot_list.append(tf.reshape(tf.slice(i_frame_in_GOP_one_hot, [0, i, 0], [-1, 1, -1]),[-1, GOP_LENGTH]))

    #----------------------------------------------------------------------------------------------------------------------------------------------------------
    # x.shape=[-1, LSTM_MAX_LENGTH, VECTOR_LENGTH = [-1, 20, 448]]
    x_64 = tf.slice(x, [0, 0, 0], [-1, LSTM_READ_LENGTH, VECTOR_LENGTH_LIST[0]])
    x_32 = tf.slice(x, [0, 0, VECTOR_LENGTH_LIST[0]], [-1, LSTM_READ_LENGTH, VECTOR_LENGTH_LIST[1]])
    x_16 = tf.slice(x, [0, 0, VECTOR_LENGTH_LIST[0]+VECTOR_LENGTH_LIST[1]], [-1, LSTM_READ_LENGTH, VECTOR_LENGTH_LIST[2]])
    
    # 2:sigmoid(x)  5:Lrelu(x) 
    acti_mode_fc = 5
    
    #lstm arguements: x, num_x_size_x, num_input_hidden_size, isdrop,  ----input, 64/128/256, 64/128/256, drop
    #    num_denlayers_2_features, num_denlayers_3_features,           ----48/96/192, 1/4/6
    #     qp_list, i_frame_in_GOP_one_hot_list, acti_mode_fc,          ----qp, GOP[4], acti
    #     variable_scope = 'RNN', name_string ='64'                    ----scope, name-string
    '''
    y_pred_flat_64_list, _ = lstm(x_64, NUM_VECTOR_SIZE_64, NUM_HIDDEN_SIZE_64, isdrop, 
         NUM_DENLAYER2_FEATURES_64, 1, 
         tf.multiply(1.0, qp_list), i_frame_in_GOP_one_hot_list, acti_mode_fc,
             variable_scope = 'RNN64', name_string ='64')   
    
    y_pred_flat_32_list, _ = lstm(x_32, NUM_VECTOR_SIZE_32, NUM_HIDDEN_SIZE_32, isdrop, 
         NUM_DENLAYER2_FEATURES_32, 4, 
         tf.multiply(1.0, qp_list), i_frame_in_GOP_one_hot_list, acti_mode_fc,
             variable_scope = 'RNN32', name_string ='32')
    
    y_pred_flat_16_list, _  = lstm(x_16, NUM_VECTOR_SIZE_16, NUM_HIDDEN_SIZE_16, isdrop,
         NUM_DENLAYER2_FEATURES_16, 16, 
         tf.multiply(1.0, qp_list), i_frame_in_GOP_one_hot_list, acti_mode_fc,
             variable_scope = 'RNN16', name_string ='16')    
    '''
    y_pred_flat_64_list, _ = attention(x_64, NUM_VECTOR_SIZE_64, NUM_HIDDEN_SIZE_64, isdrop, 
         NUM_DENLAYER2_FEATURES_64, 1, 
         tf.multiply(1.0, qp_list), i_frame_in_GOP_one_hot_list, acti_mode_fc,
             scope = 'ATTEN64', name_string ='64')   
    
    y_pred_flat_32_list, _ = attention(x_32, NUM_VECTOR_SIZE_32, NUM_HIDDEN_SIZE_32, isdrop, 
         NUM_DENLAYER2_FEATURES_32, 4, 
         tf.multiply(1.0, qp_list), i_frame_in_GOP_one_hot_list, acti_mode_fc,
             scope = 'ATTEN32', name_string ='32')
    
    y_pred_flat_16_list, _  = attention(x_16, NUM_VECTOR_SIZE_16, NUM_HIDDEN_SIZE_16, isdrop,
         NUM_DENLAYER2_FEATURES_16, 16, 
         tf.multiply(1.0, qp_list), i_frame_in_GOP_one_hot_list, acti_mode_fc,
             scope = 'ATTEN16', name_string ='16')
    
    qp_var = tf.reduce_mean(tf.square(qp-tf.reduce_mean(qp,axis=0)))
    
    y_truth_valid = tf.reshape(tf.slice(y_, [0,0,0],[-1,LSTM_OUTPUT_LENGTH,-1]),[-1, 16])
    
    y_flat_64 = tf.reshape(tf.concat(y_flat_64_list, axis=1),[-1,1])
    y_flat_32 = tf.reshape(tf.concat(y_flat_32_list, axis=1),[-1,4])
    y_flat_16 = tf.reshape(tf.concat(y_flat_16_list, axis=1),[-1,16])
    
    y_flat_valid_32 = tf.reshape(tf.concat(y_flat_valid_32_list, axis=1),[-1,4])
    y_flat_valid_16 = tf.reshape(tf.concat(y_flat_valid_16_list, axis=1),[-1,16])
    
    y_pred_flat_64 = tf.reshape(tf.concat(y_pred_flat_64_list, axis=1),[-1,1])
    y_pred_flat_32 = tf.reshape(tf.concat(y_pred_flat_32_list, axis=1),[-1,4])
    y_pred_flat_16 = tf.reshape(tf.concat(y_pred_flat_16_list, axis=1),[-1,16])
    
    #-----------------------------------------------------------------------------

    if is_balance == True:
        loss_64_mean_pos = tf.reduce_sum( - tf.multiply(y_flat_64, tf.log(y_pred_flat_64 + 1e-12))) / (tf.to_float(tf.count_nonzero(y_flat_64))+1e-12)
        loss_64_mean_neg = tf.reduce_sum( - tf.multiply((1 - y_flat_64), tf.log((1 - y_pred_flat_64) + 1e-12))) / (tf.to_float(tf.count_nonzero(1 - y_flat_64))+1e-12)
        loss_64 = (loss_64_mean_pos + loss_64_mean_neg) / 2
        
        loss_32_mean_pos = tf.reduce_sum( tf.multiply( - tf.multiply(y_flat_32, tf.log(y_pred_flat_32 + 1e-12)) , y_flat_valid_32))/ (tf.to_float(tf.count_nonzero(tf.multiply(y_flat_32, y_flat_valid_32)))+1e-12)
        loss_32_mean_neg = tf.reduce_sum( tf.multiply( - tf.multiply((1 - y_flat_32), tf.log((1 - y_pred_flat_32) + 1e-12)) , y_flat_valid_32)) / (tf.to_float(tf.count_nonzero(tf.multiply((1 - y_flat_32), y_flat_valid_32)))+1e-12)
        loss_32 = (loss_32_mean_pos + loss_32_mean_neg) / 2
        
        loss_16_mean_pos = tf.reduce_sum( tf.multiply( - tf.multiply(y_flat_16, tf.log(y_pred_flat_16 + 1e-12)) , y_flat_valid_16)) / (tf.to_float(tf.count_nonzero(tf.multiply(y_flat_16, y_flat_valid_16)))+1e-12)
        loss_16_mean_neg = tf.reduce_sum( tf.multiply( - tf.multiply((1 - y_flat_16), tf.log((1 - y_pred_flat_16) + 1e-12)) , y_flat_valid_16)) / (tf.to_float(tf.count_nonzero(tf.multiply((1 - y_flat_16), y_flat_valid_16)))+1e-12)   
        loss_16 = (loss_16_mean_pos + loss_16_mean_neg) / 2
    else:
        loss_64_pos = tf.reduce_sum( - tf.multiply(y_flat_64, tf.log(y_pred_flat_64 + 1e-12))) 
        loss_64_neg = tf.reduce_sum( - tf.multiply((1 - y_flat_64), tf.log((1 - y_pred_flat_64) + 1e-12))) 
        loss_64 = (loss_64_pos + loss_64_neg) / (tf.to_float(tf.count_nonzero(y_flat_64) + tf.count_nonzero(1 - y_flat_64))+1e-12)
        
        loss_32_pos = tf.reduce_sum( tf.multiply( - tf.multiply(y_flat_32, tf.log(y_pred_flat_32 + 1e-12)) , y_flat_valid_32)) 
        loss_32_neg = tf.reduce_sum( tf.multiply( - tf.multiply((1 - y_flat_32), tf.log((1 - y_pred_flat_32) + 1e-12)) , y_flat_valid_32)) 
        loss_32 = (loss_32_pos + loss_32_neg) / (tf.to_float(tf.count_nonzero(y_flat_valid_32))+1e-12)
        
        loss_16_pos = tf.reduce_sum( tf.multiply( - tf.multiply(y_flat_16, tf.log(y_pred_flat_16 + 1e-12)) , y_flat_valid_16)) 
        loss_16_neg = tf.reduce_sum( tf.multiply( - tf.multiply((1 - y_flat_16), tf.log((1 - y_pred_flat_16) + 1e-12)) , y_flat_valid_16)) 
        loss_16 = (loss_16_pos + loss_16_neg) / (tf.to_float(tf.count_nonzero(y_flat_valid_16))+1e-12)

    watcher_1 = y_flat_16
    watcher_2 = i_frame_in_GOP_one_hot
    watcher_3 = y_pred_flat_16
    
    total_loss = loss_16 + loss_32 + loss_64

    learning_rate_current = tf.train.exponential_decay(learning_rate_init, global_step, decay_step, decay_rate, staircase = True)  
    
    opt_vars_all = [v for v in tf.trainable_variables()]
    opt_vars = opt_vars_all
    
    net_all = [x for x in tf.global_variables()]
    #for i in range(len(net_all)):
    #    print(net_all[i])
    	         
    if limit_grad == True:
        grads, _ = tf.clip_by_global_norm(tf.gradients(total_loss, opt_vars), MAX_GRAD_NORM)
        optimizer = tf.train.MomentumOptimizer(learning_rate_current,momentum)
        train_step = optimizer.apply_gradients(zip(grads, opt_vars), global_step=tf.contrib.framework.get_or_create_global_step())    
    else:
        train_step = tf.train.MomentumOptimizer(learning_rate_current,momentum).minimize(total_loss, var_list=opt_vars)
        
    correct_prediction_valid_32 = tf.multiply(y_flat_valid_32, tf.cast(tf.equal(tf.round(y_pred_flat_32), tf.round(y_flat_32)), tf.float32))
    correct_prediction_valid_16 = tf.multiply(y_flat_valid_16, tf.cast(tf.equal(tf.round(y_pred_flat_16), tf.round(y_flat_16)), tf.float32))
    correct_prediction_64 = tf.equal(tf.round(y_pred_flat_64), tf.round(y_flat_64))
    accuracy_16 = tf.reduce_sum(tf.multiply(y_flat_valid_16, tf.cast(correct_prediction_valid_16, "float")))/(tf.reduce_sum(y_flat_valid_16)+1e-12)
    accuracy_32 = tf.reduce_sum(tf.multiply(y_flat_valid_32, tf.cast(correct_prediction_valid_32, "float")))/(tf.reduce_sum(y_flat_valid_32)+1e-12)
    accuracy_64 = tf.reduce_mean(tf.cast(correct_prediction_64, "float"))    
    
    accuracy_list = tf.stack([accuracy_64, accuracy_32, accuracy_16])
    loss_list = tf.stack([loss_64, loss_32, loss_16])
    
    # y_flat_64/32/16 is ground-truth
    # y_pred_flat_64/32/16 is prediction
    
    return y_truth_valid, y_flat_64, y_flat_32, y_flat_16, y_pred_flat_64, y_pred_flat_32, y_pred_flat_16, total_loss, loss_list, learning_rate_current, train_step, accuracy_list, opt_vars_all, watcher_1, watcher_2, watcher_3
