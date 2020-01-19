using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace ObjectDetection.DataStructures
{
    public class ImageNetPrediction
    {
        /// <summary>
        /// O ImageNetPrediction é a classe de dados de previsão e conta com os seguintes float[] campos:
        /// PredictedLabel contém as dimensões, a pontuação de objeções e as probabilidades de classe para
        /// cada uma das caixas delimitadores detectadas em uma imagem.
        /// </summary>
        [ColumnName("grid")]
        public float[] PredictedLabels;
    }
}
