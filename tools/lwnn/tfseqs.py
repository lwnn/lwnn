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