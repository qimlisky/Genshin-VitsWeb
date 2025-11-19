# Genshin-VitsWeb

快速调用 Vits ;在 Web 生成语音

## 部署指南
你可以尝试用miniconda
pip源https://mirrors.tuna.tsinghua.edu.cn/help/pypi/
```url
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

| 软件名      | 下载链接    |
| ----------- | ----------- |
| Git         | [点我下载](https://ghproxy.com/github.com/git-for-windows/git/releases/download/v2.37.3.windows.1/Git-2.37.3-64-bit.exe)       |
| Miniconda   | [点我下载](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)        |
如果你是小白 并且以上提及的几款软件你全都没装 [请看这里](install.md)
conda create -n （自己取一个例如voice） python=3.8 -y
conda activate 自己取一个例如voice一致）
git clone https://github.com/HuanLinMaster/Genshin-VitsWeb
cd （文件名，例如VitsWeb）

### Step 2 安装 Nodejs
```code
conda install -c conda-forge nodejs
```

### Step 3 克隆 Vits 仓库并下载数据集（已经下载完毕，名为vtis）

##### 注: Vits 仓库是要在本项目目录下克隆的！！（名为VitsWeb，已下载）
```
git clone --depth=1 https://github.com/Stardust-minus/vits
```
慢的话换成这个
```
git clone --depth=1 http://ghproxy.com/https://github.com/Stardust-minus/vits
```
或者这个
```
git clone --depth=1 http://gitclone.com/github.com/Stardust-minus/vits
```
自行下载 [数据集](https://obs.baimianxiao.cn/share/obs/sankagenkeshi/G_809000.pth) 并丢到vits目录内 ( ./vits （已下载）)

### Step 4 安装 Vits 依赖及 Pytorch
--- 
以下命令需要在 ./Vits 目录内执行
去掉 ./vits/requirements.txt 中的 torchvision 那一行
```requirement
Cython==0.29.21
librosa==0.8.0
matplotlib==3.3.1
numpy==1.18.5
phonemizer==2.2.1
scipy==1.5.2
tensorboard==2.3.0
# torchvision==0.7.0！！！！！（这行去掉）
Unidecode==1.1.1
pypinyin
pypinyin_dict
jieba
```
####安装依赖
```
！pip install -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com（清华也行）
！pip uninstall torch torchvision torchaudio
```

分情况讨论
> 如果你的电脑没有显卡或者你不想要用显卡
> 
> 执行
> 
> (Mac/Windows)
> ```
> pip3 install torch torchvision torchaudio 
> ```
> (Linux)
> ```
> pip3 install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple（可自行去pytorch找代码https://pytorch.org/，pip，python，cpu）
>```

> 如果你的电脑有显卡 <br>
> [在这里选择自己的显卡型号，下载安装](https://www.nvidia.cn/Download/index.aspx?lang=cn#) <br>
> [在这里下载CUDNN,需要注册](https://developer.nvidia.com/rdp/cudnn-download) <br>
> 复制 cuDNN bin 目录下的文件到 CUDA 的 bin 目录下（.dll） <br>
> 复制 cuDNN include 目录下的文件到 CUDA 的 include 目录下（.h） <br>
> 复制 cuDNN lib/x64 目录下的文件到 CUDA 的 lib/x64 目录下（.lib） <br>
> 添加环境变量，把 C:\Program Files\NVIDIA GPU Computing > Toolkit\CUDA\v10.1\lib\x64 加到 path 中 <br>

然后两种情况都需要的

安装Visual Studio2019的
 ```
MSVC
windows SDK
CMake
```

重启电脑！！！一定要重启！！！

重启完了继续编译

进入 ./vits/monotonic_align/ 目录 执行
```
python setup.py build_ext --inplace
```


### Step 5 开启服务端
---
回到项目目录
```
npm i --registry=https://registry.npmmirror.com/
node ./app.js
```
