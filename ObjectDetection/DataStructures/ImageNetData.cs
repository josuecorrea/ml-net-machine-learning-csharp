using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;

namespace ObjectDetection.DataStructures
{    public class ImageNetData
    {
        //Um ImagePath que contém o caminho no qual a imagem é armazenada.
        [LoadColumn(0)]
        public string ImagePath;

        //Um Label que contém o nome do arquivo.
        [LoadColumn(1)]
        public string Label;


        /// <summary>
        /// Além disso, ImageNetData contém um método ReadFromFile que carrega vários arquivos de imagem armazenados no 
        /// caminho imageFolder especificado e os retorna como uma coleção de objetos ImageNetData.
        /// </summary>
        /// <param name="imageFolder"></param>
        /// <returns></returns>
        public static IEnumerable<ImageNetData> ReadFromFile(string imageFolder)
        {
            return Directory
                .GetFiles(imageFolder)
                .Where(filePath => Path.GetExtension(filePath) != ".md")
                .Select(filePath => new ImageNetData { ImagePath = filePath, Label = Path.GetFileName(filePath) });
        }
    }
}
