using System.Drawing;

namespace ObjectDetection.YoloParser
{
    public class BoundingBoxDimensions : DimensionsBase { }
    public class YoloBoundingBox
    {
        //Dimensions contém as dimensões da caixa delimitadora.
        //Label contém a classe de objeto detectada na caixa delimitadora.
        // Confidence contém a confiança da classe.
        // Rect contém a representação de retângulo das dimensões da caixa delimitadora.
        // BoxColor contém a cor associada à respectiva classe usada para desenhar na imagem.

        public BoundingBoxDimensions Dimensions { get; set; }

        public string Label { get; set; }

        public float Confidence { get; set; }

        public RectangleF Rect
        {
            get { return new RectangleF(Dimensions.X, Dimensions.Y, Dimensions.Width, Dimensions.Height); }
        }

        public Color BoxColor { get; set; }
    }
}
