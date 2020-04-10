using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using saga.Models;
using System.Drawing;


namespace saga.Controllers
{
    public class HomeController : Controller
    {
        [HttpPost]
        public IActionResult Index(Consumer c)
        {
            int i = 0;
            return RedirectToAction("PredictLoanType", "Home", c);
        }
        [HttpGet]
        public IActionResult Index()
        {

            return View();
        }
        public IActionResult PredictLoanType(Consumer c)
        {
            int i = 0;

            Bitmap bmp = new Bitmap(1240, 450);
            Graphics g = Graphics.FromImage(bmp);
            g.Clear(Color.White);
            bmp.Save(@"C:\Users\milja\source\repos\saga\saga\wwwroot\images\graph.jpg", System.Drawing.Imaging.ImageFormat.Jpeg);
            Pen myPen = new Pen(Color.Black);
            g.DrawLine(myPen, 10, 50, 10, 400);
            g.DrawLine(myPen, 7, 50, 13, 50);
            g.DrawLine(myPen, 7, 137, 13, 137);
            g.DrawLine(myPen, 7, 225, 13, 225);
            g.DrawLine(myPen, 7, 312, 13, 312);
            g.DrawLine(myPen, 7, 400, 13, 400);
            g.DrawLine(myPen, 10, 400, 1230, 400);
            g.DrawLine(myPen, 10, 50, 1230, 50);
            g.DrawLine(myPen, 1230, 50, 1230, 400);
            bool again = true;
            while (again)
            {
                try
                {
                    using (System.IO.StreamWriter file =
                        new System.IO.StreamWriter(@"C:\Users\milja\source\repos\saga\saga\Output\newdata.txt", true))
                    {
                        file.WriteLine("-1" + " " + c.UserStat1);
                    }
                    again = false;
                }
                catch (System.IO.IOException) { }
            }
            System.Threading.Thread.Sleep(10000);
      //      while (new System.IO.FileInfo(@"C:\Users\milja\source\repos\saga\saga\Output\result.txt").Length == 0) { }
       /*     string[] lines = System.IO.File.ReadAllLines(@"C:\Users\milja\source\repos\saga\saga\Output\result.txt");
            int x = 40;int flag = 0;
            foreach (string line in lines)
            {
                float perc = 0;
                try
                {
                     perc = float.Parse(line);
                }catch(System.FormatException e) {
                    break; }
                SolidBrush blueBrush = new SolidBrush(Color.Black);
                Rectangle rect = new Rectangle(x, (int)(50 + (350 - (350 * perc / 100))), 200, (int)(350 * perc / 100));
                g.FillRectangle(blueBrush, rect);
                x += 240;
            }
            using (var fs = new System.IO.FileStream(@"C:\Users\milja\source\repos\saga\saga\Output\result.txt", System.IO.FileMode.Truncate)) { }
            if(flag==1)bmp.Save(@"C:\Users\milja\source\repos\saga\saga\wwwroot\images\graph.jpg", System.Drawing.Imaging.ImageFormat.Jpeg);*/
            return RedirectToAction("Index", "Home");
        }
        [HttpGet]
        public IActionResult About()
        {
            return View();
        }
        [HttpPost]
        public IActionResult About(Alg a)
        {
            return RedirectToAction("Users", "Home", a);
        }

        public IActionResult Users(Alg a)
        {
            using (System.IO.StreamWriter file =
             new System.IO.StreamWriter(@"C:\Users\milja\source\repos\saga\saga\Output\newdata.txt", true))
            {
                file.WriteLine("-2" + " " + (int)a.Type);
            }
            System.Threading.Thread.Sleep(10000);
            while (new System.IO.FileInfo(@"C:\Users\milja\source\repos\saga\saga\Output\stanbeni.txt").Length == 0) { }
            System.Collections.Generic.List<saga.Models.Consumer> model = new System.Collections.Generic.List<saga.Models.Consumer>();
            string[] lines = System.IO.File.ReadAllLines(@"C:\Users\milja\source\repos\saga\saga\Output\stanbeni.txt");
            int x = 40;
            int i = 0;
            foreach (string line in lines)
            {
                string aa = ((i <= 9) ?"stanbeni" : "gotovinski");
                i++;
                string perc = line;
                string[] split = perc.Split(' ');
                int float0 = int.Parse(split[0]);
                float float1 = float.Parse(split[1]);
                Consumer c=new Consumer(float0, float1,aa ,0,0,0);
                model.Add(c);
            }
            bool again = true;
            while(again)
           try
           {
             using (var fs = new System.IO.FileStream(@"C:\Users\milja\source\repos\saga\saga\Output\stanbeni.txt", System.IO.FileMode.Truncate)) { }
                   again = false;
               }
            catch (System.IO.IOException e) { }
            return View(model);
        }
        public IActionResult Contact()
        {
            ViewData["Message"] = "Your contact page.";

            return View();
        }

        public IActionResult Privacy()
        {
            return View();
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }

    }
}
