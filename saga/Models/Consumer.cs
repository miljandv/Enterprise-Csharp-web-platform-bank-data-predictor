using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;

namespace saga.Models
{
    public class Consumer
    {
        public Consumer()
        {
            UserStat1 = 0;
            UserStat2 = 0;
            UserStat3 = "";
            UserStat4 = 0;
            UserStat5 = 0;
            UserStat6 = 0;

        }
        public Consumer(int s1,float s2,string s3,float s4,float s5,float s6)
        {
            UserStat1 = s1;
            UserStat2 = s2;
            UserStat3 = s3;
            UserStat4 = s4;
            UserStat5 = s5;
            UserStat6 = s6;
        }
        [DisplayName("First Name")]
        public int UserStat1 { get; set; }
        [DisplayName("First Name")]
        public float UserStat2 { get; set; }
        [DisplayName("First Name")]
        public string UserStat3 { get; set; }
        [DisplayName("First Name")]
        public float UserStat4 { get; set; }
        [DisplayName("First Name")]
        public float UserStat5 { get; set; }
        [DisplayName("First Name")]
        public float UserStat6 { get; set; }

        public void ExecutePython(string pathtoscript)
        {
            
            ProcessStartInfo start = new ProcessStartInfo();
            start.FileName = @"C:\Users\milja\AppData\Local\Programs\Python\Python36\python.exe";//cmd is full path to python.exe
            start.Arguments = pathtoscript;//args is path to .py file and any cmd line args
            start.UseShellExecute = false;
            start.RedirectStandardOutput = true;
            using (Process process = Process.Start(start))
            {
                using (StreamReader reader = process.StandardOutput)
                {
                    string result = reader.ReadToEnd();
                    Console.Write(result);
                }
            }
            
        }
    }
}


