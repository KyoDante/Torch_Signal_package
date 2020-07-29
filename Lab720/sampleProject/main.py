
# ref: 
# 有关import的事项
# https://www.cnblogs.com/liuyanhang/p/11018407.html
# https://zhuanlan.zhihu.com/p/64893308
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(os.path.split(curPath)[0])[0] # 嵌套了两层，需要套娃
sys.path.append(rootPath)

import utils.audio.workflow

if __name__ == "__main__":
    print("Lab720")