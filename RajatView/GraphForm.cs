using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Numerics;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.Util;
using Emgu.CV.CvEnum;
using Emgu.CV.ML;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using ZedGraph;

namespace RajatView
{
    public partial class GraphForm : Form
    {
        public Image<Gray, byte> img = null;
        public GraphForm()
        {
            InitializeComponent();
        }

        public void setImage(Image<Gray, byte> img)
        {
            this.img = img;
        }

        public void CreateGraph(ZedGraphControl zgc)
        {
            GraphPane myPane = zgc.GraphPane;

            myPane.Title.Text = "Image Histogram";
            myPane.XAxis.Title.Text = "Intensity";
            myPane.YAxis.Title.Text = "Frequency";

            double[] y = new double[256];
            
            PointPairList list1 = new PointPairList();
            Image<Gray, byte> imgs = img;

            for (int i = 0; i < imgs.Height; i++)
                for (int j = 0; j < imgs.Width; j++)
                {
                    y[(int)imgs[i, j].Intensity]++;
                }
            for (int i = 0; i < 255; i++)
            {
                list1.Add(i, y[i]);
            }

            LineItem myCurve = myPane.AddCurve("My Curve", list1, Color.Blue,
                                    SymbolType.Circle);

            myCurve.Line.Fill = new Fill(Color.White, Color.Red, 45F);
            
            myCurve.Symbol.Fill = new Fill(Color.White);
            
            myPane.Chart.Fill = new Fill(Color.White, Color.LightGoldenrodYellow, 45F);
            
            myPane.Fill = new Fill(Color.White, Color.FromArgb(220, 220, 255), 45F);

            zgc.AxisChange();
        }
        public void Form2_Resize(object sender, EventArgs e)
        {
            SetSize();
        }

        public void SetSize()
        {
            zedGraphControl1.Location = new Point(10, 10);
            zedGraphControl1.Size = new Size(this.ClientRectangle.Width - 20, this.ClientRectangle.Height - 20);
        }

        private void Form2_Load(object sender, EventArgs e)
        {
            zedGraphControl1.Visible = true;
            CreateGraph(zedGraphControl1);
            SetSize();
        }

        
    }
}
