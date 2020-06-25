TFSEQ_BATCHNORM_COND = {
 'Connection': {0: [1, 3],
                1: [23, 18],
                2: [23, 18],
                3: [2, 12, 5],
                4: [39, 18],
                5: [4, 6],
                6: [9, 10],
                7: [21, 18],
                8: [39, 18],
                9: [8, 7],
                10: [19],
                11: [41, 18],
                12: [11, 13],
                13: [16, 17],
                14: [22, 18],
                15: [41, 18],
                16: [15, 14],
                17: [19],
                18: [47],
                19: [20],
                20: [47, 47],
                21: [29, 33],
                22: [29, 33],
                23: [29, 33],
                24: [39, 36],
                25: [41, 36],
                26: [45, 36],
                27: [43, 36],
                28: [48, 36],
                29: [28, 27, 26, 25, 24],
                30: [45, 36],
                31: [43, 36],
                32: [48, 36],
                33: [32, 31, 30, 35, 34],
                34: [37],
                35: [37],
                36: [47],
                37: [38],
                38: [47, 47],
                39: [40],
                40: [],
                41: [42],
                42: [],
                43: [44],
                44: [],
                45: [46],
                46: [],
                47: [],
                48: []},
 'Sequence': {0: 'Merge', # cond_1/Merge',
              1: 'Switch', # cond_1/Switch_1',
              2: 'Switch', # cond_1/Identity/Switch',
              3: 'Identity', # cond_1/Identity',
              4: 'Switch', # '
                 # cond_1/AssignMovingAvg_1/Switch',
              5: 'Sub', # cond_1/AssignMovingAvg_1',
              6: 'Mul', # cond_1/AssignMovingAvg_1/mul',
              7: 'Switch', # '
                 # cond_1/AssignMovingAvg_1/sub/Switch_1',
              8: 'Switch', # '
                 # cond_1/AssignMovingAvg_1/sub/Switch',
              9: 'Sub', # cond_1/AssignMovingAvg_1/sub',
              10: 'Constant', # '
                  # cond_1/AssignMovingAvg_1/decay',
              11: 'Switch', # '
                  # cond_1/AssignMovingAvg/Switch',
              12: 'Sub', # cond_1/AssignMovingAvg',
              13: 'Mul', # cond_1/AssignMovingAvg/mul',
              14: 'Switch', # '
                  # cond_1/AssignMovingAvg/sub/Switch_1',
              15: 'Switch', # '
                  # cond_1/AssignMovingAvg/sub/Switch',
              16: 'Sub', # cond_1/AssignMovingAvg/sub',
              17: 'Constant', # '
                  # cond_1/AssignMovingAvg/decay',
              18: 'Identity', # cond_1/pred_id',
              19: 'Identity', # cond_1/switch_t',
              20: 'Switch', # cond_1/Switch',
              21: 'Merge', # cond/Merge_2',
              22: 'Merge', # cond/Merge_1',
              23: 'Merge', # cond/Merge',
              24: 'Switch', # '
                  # cond/FusedBatchNorm_1/Switch_4',
              25: 'Switch', # '
                  # cond/FusedBatchNorm_1/Switch_3',
              26: 'Switch', # '
                  # cond/FusedBatchNorm_1/Switch_2',
              27: 'Switch', # '
                  # cond/FusedBatchNorm_1/Switch_1',
              28: 'Switch', # '
                  # cond/FusedBatchNorm_1/Switch',
              29: 'BatchNormalization', # '
                  # cond/FusedBatchNorm_1',
              30: 'Switch', # '
                  # cond/FusedBatchNorm/Switch_2',
              31: 'Switch', # '
                  # cond/FusedBatchNorm/Switch_1',
              32: 'Switch', # '
                  # cond/FusedBatchNorm/Switch',
              33: 'BatchNormalization', # '
                  # cond/FusedBatchNorm',
              34: 'Constant', # cond/Const_1',
              35: 'Constant', # cond/Const',
              36: 'Identity', # cond/pred_id',
              37: 'Identity', # cond/switch_t',
              38: 'Switch', # cond/Switch',
              39: 'Identity', # moving_variance/read',
              40: 'Constant', # moving_variance',
              41: 'Identity', # moving_mean/read',
              42: 'Constant', # moving_mean',
              43: 'Identity', # gamma/read',
              44: 'Constant', # gamma',
              45: 'Identity', # beta/read',
              46: 'Constant', # beta',
              47: '?', # learning_params/Placeholder_1',
              48: '?', # X'
              }}


TFSEQ_LAYER_NORM = {
 'Connection': {0: [3, 1],
                1: [16, 2],
                2: [12, 4],
                3: [18, 4],
                4: [5, 14],
                5: [6],
                6: [8, 7],
                7: [],
                8: [10, 9],
                9: [],
                10: [18, 11],
                11: [12],
                12: [18, 13],
                13: [],
                14: [15],
                15: [],
                16: [17],
                17: [],
                18: []},
 'Sequence': {0: 'Add', # LayerNorm/batchnorm/add_1',
              1: 'Sub', # LayerNorm/batchnorm/sub',
              2: 'Mul', # LayerNorm/batchnorm/mul_2',
              3: 'Mul', # LayerNorm/batchnorm/mul_1',
              4: 'Mul', # LayerNorm/batchnorm/mul',
              5: 'Rsqrt', # LayerNorm/batchnorm/Rsqrt',
              6: 'Add', # LayerNorm/batchnorm/add',
              7: 'Constant', # LayerNorm/batchnorm/add/y',
              8: 'Mean', # LayerNorm/moments/variance',
              9: 'Constant', # '
                 #'LayerNorm/moments/variance/reduction_indices',
              10: 'SquaredDifference', # '
                  #'LayerNorm/moments/SquaredDifference',
              11: 'StopGradient', # '
                  #'LayerNorm/moments/StopGradient',
              12: 'Mean', # LayerNorm/moments/mean',
              13: 'Constant', # '
                  #'LayerNorm/moments/mean/reduction_indices',
              14: 'Identity', # LayerNorm/gamma/read',
              15: 'Constant', # LayerNorm/gamma',
              16: 'Identity', # LayerNorm/beta/read',
              17: 'Constant', # LayerNorm/beta',
              18: '?', # input'
    }
}

TFSEQ_EINSUM1 = {
 'Connection': {0: [1, 20],
                1: [4, 2],
                2: [15, 11, 3],
                3: [],
                4: [5, 21],
                5: [22, 6],
                6: [8, 7],
                7: [],
                8: [9, 11],
                9: [10, 15],
                10: [],
                11: [19, 14, 13, 12],
                12: [],
                13: [],
                14: [],
                15: [19, 18, 17, 16],
                16: [],
                17: [],
                18: [],
                19: [22],
                20: [],
                21: [],
                22: []},
 'Sequence': {0: 'Add', # add',
              1: 'Reshape', # '
                 # einsum/Reshape_1',
              2: 'Pack', # '
                 # einsum/Reshape_1/shape',
              3: 'Constant', # '
                 # einsum/Reshape_1/shape/2',
              4: 'MatMul', # '
                 # einsum/MatMul',
              5: 'Reshape', # '
                 # einsum/Reshape',
              6: 'Pack', # '
                 # einsum/Reshape/shape',
              7: 'Constant', # '
                 # einsum/Reshape/shape/1',
              8: 'Mul', # '
                 # einsum/mul_1',
              9: 'Mul', # einsum/mul',
              10: 'Constant', # '
                  # einsum/mul/x',
              11: 'StridedSlice', # '
                  # einsum/strided_slice_1',
              12: 'Constant', # '
                  # einsum/strided_slice_1/stack_2',
              13: 'Constant', # '
                  # einsum/strided_slice_1/stack_1',
              14: 'Constant', # '
                  # einsum/strided_slice_1/stack',
              15: 'StridedSlice', # '
                  # einsum/strided_slice',
              16: 'Constant', # '
                  # einsum/strided_slice/stack_2',
              17: 'Constant', # '
                  # einsum/strided_slice/stack_1',
              18: 'Constant', # '
                  # einsum/strided_slice/stack',
              19: 'Shape', # '
                  # einsum/Shape',
              20: '?', # bias/read',
              21: '?', # kernel/read',
              22: '?', # input'
}}

TFSEQ_EINSUM2 = {
 'Connection': {0: [1, 27],
                1: [3, 2],
                2: [],
                3: [7, 4],
                4: [20, 16, 6, 5],
                5: [],
                6: [],
                7: [10, 8],
                8: [25, 9],
                9: [],
                10: [28, 11],
                11: [13, 12],
                12: [],
                13: [14, 16],
                14: [15, 20],
                15: [],
                16: [24, 19, 18, 17],
                17: [],
                18: [],
                19: [],
                20: [24, 23, 22, 21],
                21: [],
                22: [],
                23: [],
                24: [28],
                25: [29, 26],
                26: [],
                27: [],
                28: [],
                29: []},
 'Sequence': {0: 'Add', # '
                 # add',
              1: 'Transpose', # '
                 # einsum/transpose_1',
              2: 'Constant', # '
                 # einsum/transpose_1/perm',
              3: 'Reshape', # '
                 # einsum/Reshape_2',
              4: 'Pack', # '
                 # einsum/Reshape_2/shape',
              5: 'Constant', # '
                 # einsum/Reshape_2/shape/3',
              6: 'Constant', # '
                 # einsum/Reshape_2/shape/2',
              7: 'MatMul', # '
                 # einsum/MatMul',
              8: 'Reshape', # '
                 # einsum/Reshape_1', weights
              9: 'Constant', # '
                 # einsum/Reshape_1/shape',
              10: 'Reshape', # '
                  # einsum/Reshape',
              11: 'Pack', # '
                  # einsum/Reshape/shape',
              12: 'Constant', # '
                  # einsum/Reshape/shape/1',
              13: 'Mul', # '
                  # einsum/mul_1',
              14: 'Mul', # '
                  # einsum/mul',
              15: 'Constant', # '
                  # einsum/mul/x',
              16: 'StridedSlice', # '
                  # einsum/strided_slice_1',
              17: 'Constant', # '
                  # einsum/strided_slice_1/stack_2',
              18: 'Constant', # '
                  # einsum/strided_slice_1/stack_1',
              19: 'Constant', # '
                  # einsum/strided_slice_1/stack',
              20: 'StridedSlice', # '
                  # einsum/strided_slice',
              21: 'Constant', # '
                  # einsum/strided_slice/stack_2',
              22: 'Constant', # '
                  # einsum/strided_slice/stack_1',
              23: 'Constant', # '
                  # einsum/strided_slice/stack',
              24: 'Shape', # '
                  # einsum/Shape',
              25: 'Transpose', # '
                  # einsum/transpose',
              26: 'Constant', # '
                  # einsum/transpose/perm',
              27: '?', # Reshape_1 bias
              28: '?', # input',
              29: '?', # Reshape'
}}
