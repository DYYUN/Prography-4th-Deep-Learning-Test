# Prography 4th Deep Learning Test
Image Classification with own dataset

윈도우에서 최신 버전의 Anaconda와 CUDA를 설치하고 pytorch, matplotlib을 설치한 뒤 동작시켰습니다.

구체적으로 Anaconda prompt에서 다음 명령어를 입력합니다.

<pre><code>$ conda create -y -n pytorch ipykernel
$ activate pytorch
$ conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
$ pip install matplotlib
</code></pre>

그 뒤, models.py, train.py, test.py와 data가 들어있는 dataset 폴더를 한 폴더에 넣습니다.

Anaconda prompt로 그 폴더에 접근한 뒤 다음 명령어를 입력하면 코드가 실행됩니다.

<pre><code>
python train.py dataset/train
python test.py dataset/test mymodel.pth
</code></pre>

train.py의 결과는 output 파일이 overload 되지 않도록 mymodel.pth가 아닌 model_new.pth로 저장되게 해두었습니다.

train 시간은 제 환경 기준으로 (GTX1050 2GB, CUDA 10) 대략 40분 정도 소요됩니다. 빠른 실행 하시려면 train.py의 epoch 수를 줄이면 됩니다.

모든 환경과 library가 처음 접해보는거라 세팅에 시간이 오래 걸려 코드를 충분히 이해하지 못했습니다. 주석이 충분치 않은 점 양해 부탁드립니다.

모든 코드는 https://github.com/jinfagang/pytorch_image_classifier 를 참조했습니다.

사실상 알고리즘은 동일하며, 과제 요건에 맞도록 파일 구성과 버전에 맞지 않는 명령어들, 몇몇 dimension 문제들을 수정하였습니다.

git도 처음이라 reference를 어떻게 걸어야 할 지도 모르겠네요. 문제 된다면 연락주시면 감사하겠습니다.
