using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace ObjectDetection.YoloParser
{
    class CellDimensions : DimensionsBase { }
    class YoloOutputParser
    {
        //ROW_COUNT é o número de linhas na grade em que a imagem é dividida.
        //COL_COUNT é o número de colunas na grade em que a imagem é dividida.
        //CHANNEL_COUNT é o número total de valores contidos em uma célula da grade.
        //BOXES_PER_CELL é o número de caixas delimitadoras em uma célula.
        //BOX_INFO_FEATURE_COUNT é o número de recursos contidos em uma caixa (x, y, altura, largura, confiança).
        //CLASS_COUNT é o número de previsões de classe contidas em cada caixa delimitadora.
        //CELL_WIDTH é a largura de uma célula na grade de imagens.
        //CELL_HEIGHT é a altura de uma célula na grade de imagens.
        //channelStride é a posição inicial da célula atual na grade.

        public const int ROW_COUNT = 13;
        public const int COL_COUNT = 13;
        public const int CHANNEL_COUNT = 125;
        public const int BOXES_PER_CELL = 5;
        public const int BOX_INFO_FEATURE_COUNT = 5;
        public const int CLASS_COUNT = 20;
        public const float CELL_WIDTH = 32;
        public const float CELL_HEIGHT = 32;

        private int channelStride = ROW_COUNT * COL_COUNT;

        private float[] anchors = new float[]
        {
            1.08F, 1.19F, 3.42F, 4.41F, 6.63F, 11.38F, 9.42F, 5.11F, 16.62F, 10.52F
        };

        private string[] labels = new string[]
        {
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        };

        private static Color[] classColors = new Color[]
        {
            Color.Khaki,
            Color.Fuchsia,
            Color.Silver,
            Color.RoyalBlue,
            Color.Green,
            Color.DarkOrange,
            Color.Purple,
            Color.Gold,
            Color.Red,
            Color.Aquamarine,
            Color.Lime,
            Color.AliceBlue,
            Color.Sienna,
            Color.Orchid,
            Color.Tan,
            Color.LightPink,
            Color.Yellow,
            Color.HotPink,
            Color.OliveDrab,
            Color.SandyBrown,
            Color.DarkTurquoise
        };

        //Sigmoid: aplica a função sigmoide que gera um número entre 0 e 1.
        //Softmax: normaliza um vetor de entrada em uma distribuição de probabilidade.
        //GetOffset: mapeia elementos na saída de um modelo unidimensional para a posição correspondente em um tensor 125 x 13 x 13.
        //ExtractBoundingBoxes: extrai as dimensões da caixa delimitadora usando o método GetOffset da saída do modelo.
        //GetConfidence extrai o valor de confiança que informa como a certeza do modelo é que ele detectou um objeto e usa a função Sigmoid para transformá-lo em uma porcentagem.
        //MapBoundingBoxToCell: usa as dimensões da caixa delimitadora e as mapeia para sua respectiva célula na imagem.
        //ExtractClasses: extrai as previsões de classe para a caixa delimitadora da saída do modelo usando o método GetOffset e as transforma em uma distribuição de probabilidade usando o método Softmax.
        //GetTopResult: seleciona a classe na lista de classes previstas com a maior probabilidade.
        //IntersectionOverUnion: filtra as caixas delimitadoras sobrepostas com probabilidades inferiores.
        private float Sigmoid(float value)
        {
            var k = (float)Math.Exp(value);
            return k / (1.0f + k);
        }

        private float[] Softmax(float[] values)
        {
            var maxVal = values.Max();
            var exp = values.Select(v => Math.Exp(v - maxVal));
            var sumExp = exp.Sum();

            return exp.Select(v => (float)(v / sumExp)).ToArray();
        }

        private int GetOffset(int x, int y, int channel)
        {
            // YOLO outputs a tensor that has a shape of 125x13x13, which 
            // WinML flattens into a 1D array.  To access a specific channel 
            // for a given (x,y) cell position, we need to calculate an offset
            // into the array
            return (channel * this.channelStride) + (y * COL_COUNT) + x;
        }

        private BoundingBoxDimensions ExtractBoundingBoxDimensions(float[] modelOutput, int x, int y, int channel)
        {
            return new BoundingBoxDimensions
            {
                X = modelOutput[GetOffset(x, y, channel)],
                Y = modelOutput[GetOffset(x, y, channel + 1)],
                Width = modelOutput[GetOffset(x, y, channel + 2)],
                Height = modelOutput[GetOffset(x, y, channel + 3)]
            };
        }

        private float GetConfidence(float[] modelOutput, int x, int y, int channel)
        {
            return Sigmoid(modelOutput[GetOffset(x, y, channel + 4)]);
        }

        private CellDimensions MapBoundingBoxToCell(int x, int y, int box, BoundingBoxDimensions boxDimensions)
        {
            return new CellDimensions
            {
                X = ((float)x + Sigmoid(boxDimensions.X)) * CELL_WIDTH,
                Y = ((float)y + Sigmoid(boxDimensions.Y)) * CELL_HEIGHT,
                Width = (float)Math.Exp(boxDimensions.Width) * CELL_WIDTH * anchors[box * 2],
                Height = (float)Math.Exp(boxDimensions.Height) * CELL_HEIGHT * anchors[box * 2 + 1],
            };
        }

        public float[] ExtractClasses(float[] modelOutput, int x, int y, int channel)
        {
            float[] predictedClasses = new float[CLASS_COUNT];
            int predictedClassOffset = channel + BOX_INFO_FEATURE_COUNT;
            for (int predictedClass = 0; predictedClass < CLASS_COUNT; predictedClass++)
            {
                predictedClasses[predictedClass] = modelOutput[GetOffset(x, y, predictedClass + predictedClassOffset)];
            }
            return Softmax(predictedClasses);
        }

        private ValueTuple<int, float> GetTopResult(float[] predictedClasses)
        {
            return predictedClasses
                .Select((predictedClass, index) => (Index: index, Value: predictedClass))
                .OrderByDescending(result => result.Value)
                .First();
        }

        private float IntersectionOverUnion(RectangleF boundingBoxA, RectangleF boundingBoxB)
        {
            var areaA = boundingBoxA.Width * boundingBoxA.Height;

            if (areaA <= 0)
                return 0;

            var areaB = boundingBoxB.Width * boundingBoxB.Height;

            if (areaB <= 0)
                return 0;

            var minX = Math.Max(boundingBoxA.Left, boundingBoxB.Left);
            var minY = Math.Max(boundingBoxA.Top, boundingBoxB.Top);
            var maxX = Math.Min(boundingBoxA.Right, boundingBoxB.Right);
            var maxY = Math.Min(boundingBoxA.Bottom, boundingBoxB.Bottom);

            var intersectionArea = Math.Max(maxY - minY, 0) * Math.Max(maxX - minX, 0);

            return intersectionArea / (areaA + areaB - intersectionArea);
        }

        public IList<YoloBoundingBox> ParseOutputs(float[] yoloModelOutputs, float threshold = .3F)
        {
            var boxes = new List<YoloBoundingBox>();

            for (int row = 0; row < ROW_COUNT; row++)
            {
                for (int column = 0; column < COL_COUNT; column++)
                {
                    for (int box = 0; box < BOXES_PER_CELL; box++)
                    {
                        //Dentro do loop mais interno, calcule a posição inicial da caixa atual na saída de um modelo unidimensional.
                        var channel = (box * (CLASS_COUNT + BOX_INFO_FEATURE_COUNT));

                        //Diretamente abaixo disso, use o método ExtractBoundingBoxDimensions para obter as dimensões da caixa delimitadora atual.
                        BoundingBoxDimensions boundingBoxDimensions = ExtractBoundingBoxDimensions(yoloModelOutputs, row, column, channel);

                        //Em seguida, use o método GetConfidence para obter a confiança para a caixa delimitadora atual.
                        float confidence = GetConfidence(yoloModelOutputs, row, column, channel);

                        //Depois disso, use o método MapBoundingBoxToCell para mapear a caixa delimitadora atual para a célula atual que está sendo processada.
                        CellDimensions mappedBoundingBox = MapBoundingBoxToCell(row, column, box, boundingBoxDimensions);

                        //Antes de efetuar qualquer processamento adicional, verifique se seu valor de confiança é maior que o limite fornecido.
                        //Se não for, processe a próxima caixa delimitadora.
                        if (confidence < threshold)
                            continue;


                        //Caso contrário, continue processando a saída.A próxima etapa é obter a distribuição de probabilidade das classes previstas
                        //para a caixa delimitadora atual usando o método ExtractClasses.
                        float[] predictedClasses = ExtractClasses(yoloModelOutputs, row, column, channel);


                        //Em seguida, use o método GetTopResult para obter o valor e o índice da classe com a maior probabilidade para a caixa atual e computar sua pontuação.
                        var (topResultIndex, topResultScore) = GetTopResult(predictedClasses);
                        var topScore = topResultScore * confidence;

                        //Use o topScore para novamente manter somente as caixas delimitadoras que estão acima do limite especificado.
                        if (topScore < threshold)
                            continue;

                        //Por fim, se a caixa delimitadora atual exceder o limite, crie um objeto BoundingBox e adicione-o à lista boxes.
                        boxes.Add(new YoloBoundingBox()
                        {
                            Dimensions = new BoundingBoxDimensions
                            {
                                X = (mappedBoundingBox.X - mappedBoundingBox.Width / 2),
                                Y = (mappedBoundingBox.Y - mappedBoundingBox.Height / 2),
                                Width = mappedBoundingBox.Width,
                                Height = mappedBoundingBox.Height,
                            },
                            Confidence = topScore,
                            Label = labels[topResultIndex],
                            BoxColor = classColors[topResultIndex]
                        });

                    }
                }
            }

            return boxes;


        }

        public IList<YoloBoundingBox> FilterBoundingBoxes(IList<YoloBoundingBox> boxes, int limit, float threshold)
        {
            //No método FilterBoundingBoxes, comece criando uma matriz igual ao tamanho das caixas detectadas e marcando todos os slots como ativos ou prontos para processamento.
            var activeCount = boxes.Count;
            var isActiveBoxes = new bool[boxes.Count];

            for (int i = 0; i < isActiveBoxes.Length; i++)
                isActiveBoxes[i] = true;

            //Em seguida, classifique a lista que contém as caixas delimitadoras em ordem decrescente com base em confiança.
            var sortedBoxes = boxes.Select((b, i) => new { Box = b, Index = i })
                    .OrderByDescending(b => b.Box.Confidence)
                    .ToList();

            var results = new List<YoloBoundingBox>();

            for (int i = 0; i < boxes.Count; i++)
            {
                if (isActiveBoxes[i])
                {
                    var boxA = sortedBoxes[i].Box;
                    results.Add(boxA);

                    if (results.Count >= limit)
                        break;

                    for (var j = i + 1; j < boxes.Count; j++)
                    {
                        if (isActiveBoxes[j])
                        {
                            var boxB = sortedBoxes[j].Box;

                            if (IntersectionOverUnion(boxA.Rect, boxB.Rect) > threshold)
                            {
                                isActiveBoxes[j] = false;
                                activeCount--;

                                if (activeCount <= 0)
                                    break;
                            }
                        }
                    }

                    if (activeCount <= 0)
                        break;
                }
            }

            return results;
        }
    }
}
