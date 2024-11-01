# **Repositório para Identificação da Câmera**
## **Calibração da Câmera**

- arquivo: CalibrateCameraPhotos.py --> Gera a matriz de transformação do plano de imagem para o plano real, baseado nas imagens (fotos) presentes na pasta [ImagemCameraCalibration](https://github.com/reginaldocardoso/CameraCalibrationArUco/tree/main/ImagemCameraCalibration). Para calibrar a sua câmera tirar fotos com ela e não usar estas;
  
- O Tabuleiro de xadrez pode ser baixado [Aqui](https://github.com/opencv/opencv/blob/4.x/doc/pattern.png) ou o tabuleiro de circulos [Aqui](https://github.com/opencv/opencv/blob/4.x/doc/acircles_pattern.png);
  
- Para calibrar a camera com video em tempo real (intenda como rodar o programa e mostrar o tabuleiro para a câmera). Abra a pasta [VideoCalibration](https://github.com/reginaldocardoso/CameraCalibrationArUco/tree/main/VideoCameraCalibration) e execute o arquivo: CameraCalibration.py com o tabulerio de xadrez em mãos;

- A calibração irá gerar um arquivo ".yaml" (neste caso foi "PhotoCalibration.yaml"), que será usado na identicação do marcador ArUco

  ## **Gerando o marcador ArUco**

  -  Na pasta []() possui alguns ArUco já gerados, mas caso queirar outros usar o seguinte comando no terminal:
```bash
python ArUco_generate.py --id 25 --type DICT_5X5_100 --output GeraChessBoard/DICT_5X5_100_id25.png 
```
Irá gerar o ArUco com a ID25;

## **Pose Estimation**
- Como o ArUco impresso executar o arquivo: SingleArucoPoseEstimation.py e movimentar;
- O código irá salvar os valores de X,y,z,roll,pitch,yaw na pasta [Dados](https://github.com/reginaldocardoso/CameraCalibrationArUco/tree/main/Dados) em arquivos no formato txt;
- O arquivo Grafico.py irá plotar a posição (xyz) salva nos arquivos txt. Na minha câmera saiu muito ruído, passei um filtro passa baixa, melhorou mas ainda ficou ruim;
