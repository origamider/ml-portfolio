import matplotlib.pyplot as plt
import numpy as np

# 学習結果の表示
def show_history(history,plt_title):
    history = np.array(history)
    fig,axes = plt.subplots(1,2)
    plt.title(plt_title)
    axes[0].set_title('学習曲線(損失)')
    axes[0].plot(history[:,0],history[:,1],c='b',label='訓練用データ')
    axes[0].plot(history[:,0],history[:,3],c='r',label='検証用データ')
    axes[0].set_xlabel('繰り返し数')
    axes[0].set_ylabel('損失')
    axes[0].legend()

    axes[1].set_title('学習曲線(精度)')
    axes[1].plot(history[:,0],history[:,2],c='b',label='訓練用データ')
    axes[1].plot(history[:,0],history[:,4],c='r',label='検証用データ')
    axes[1].set_xlabel('繰り返し数')
    axes[1].set_ylabel('精度')
    axes[1].legend()
    plt.show()