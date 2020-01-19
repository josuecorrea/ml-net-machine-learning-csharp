
namespace ObjectDetection.YoloParser
{
    public class DimensionsBase
    {
        /// <summary>
        /// X contém a posição do objeto ao longo do eixo x.
        ///Y contém a posição do objeto ao longo do eixo y.
        ///Height contém a altura do objeto.
        ///Width contém a largura do objeto.
        /// </summary>
        public float X { get; set; }
        public float Y { get; set; }
        public float Height { get; set; }
        public float Width { get; set; }
    }
}
