#pragma checksum "C:\Users\milja\source\repos\saga\saga\Views\Home\Users.cshtml" "{ff1816ec-aa5e-4d10-87f7-6f4963833460}" "a63878aef805a0345e1b910bb8f4663bab5ef139"
// <auto-generated/>
#pragma warning disable 1591
[assembly: global::Microsoft.AspNetCore.Razor.Hosting.RazorCompiledItemAttribute(typeof(AspNetCore.Views_Home_Users), @"mvc.1.0.view", @"/Views/Home/Users.cshtml")]
[assembly:global::Microsoft.AspNetCore.Mvc.Razor.Compilation.RazorViewAttribute(@"/Views/Home/Users.cshtml", typeof(AspNetCore.Views_Home_Users))]
namespace AspNetCore
{
    #line hidden
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Threading.Tasks;
    using Microsoft.AspNetCore.Mvc;
    using Microsoft.AspNetCore.Mvc.Rendering;
    using Microsoft.AspNetCore.Mvc.ViewFeatures;
#line 1 "C:\Users\milja\source\repos\saga\saga\Views\_ViewImports.cshtml"
using saga;

#line default
#line hidden
#line 1 "C:\Users\milja\source\repos\saga\saga\Views\Home\Users.cshtml"
using saga.Models;

#line default
#line hidden
    [global::Microsoft.AspNetCore.Razor.Hosting.RazorSourceChecksumAttribute(@"SHA1", @"a63878aef805a0345e1b910bb8f4663bab5ef139", @"/Views/Home/Users.cshtml")]
    [global::Microsoft.AspNetCore.Razor.Hosting.RazorSourceChecksumAttribute(@"SHA1", @"fb593101479f1da9ea1ebd6c39f2e447ffc5a30c", @"/Views/_ViewImports.cshtml")]
    public class Views_Home_Users : global::Microsoft.AspNetCore.Mvc.Razor.RazorPage<System.Collections.Generic.List<saga.Models.Consumer>>
    {
        #pragma warning disable 1998
        public async override global::System.Threading.Tasks.Task ExecuteAsync()
        {
            BeginContext(82, 902, true);
            WriteLiteral(@"<br />
<br />
<br />
<style>

#tablehead {
    text-align: center;
    font-size: 20px;
    padding: 0 0 13px 0;
}
#tbl {
    text-align: left;
    background: rgba(240, 240, 240, 0.8);
    padding: 8px 6px 0 10px;
}
#tabl {
    font-family: ""Trebuchet MS"", Arial, Helvetica, sans-serif;
    border-collapse: collapse;
    width: 100%;
}

    #tabl td, #tabl th {
        border: 1px solid #ddd;
        padding: 8px;
    }

    #tabl tr:nth-child(even) {
        background-color: #f2f2f2;
    }

    #tabl tr:hover {
        background-color: #ddd;
    }

    #tabl th {
        padding-top: 12px;
        padding-bottom: 12px;
        text-align: left;
        background-color: #FFF200;
        color: black;
    }
table a:hover {
    -webkit-transform: scale(5);
}

table {
    font-family: ""Trebuchet MS"", Arial, Helvetica, sans-serif;
}
</style>
");
            EndContext();
#line 52 "C:\Users\milja\source\repos\saga\saga\Views\Home\Users.cshtml"
 if (Model != null && Model.Count != 0)
{

#line default
#line hidden
            BeginContext(1028, 164, true);
            WriteLiteral("    <table id=\"tabl\">\r\n        <tr>\r\n            <th>ID</th>\r\n            <th>Verovatnoca uzimanja kredita</th>\r\n            <th>Tip kredita</th>\r\n\r\n        </tr>\r\n");
            EndContext();
#line 61 "C:\Users\milja\source\repos\saga\saga\Views\Home\Users.cshtml"
         for (int i = 0; i < Model.Count; i++)
        {

#line default
#line hidden
            BeginContext(1251, 38, true);
            WriteLiteral("            <tr>\r\n                <td>");
            EndContext();
            BeginContext(1290, 18, false);
#line 64 "C:\Users\milja\source\repos\saga\saga\Views\Home\Users.cshtml"
               Write(Model[i].UserStat1);

#line default
#line hidden
            EndContext();
            BeginContext(1308, 27, true);
            WriteLiteral("</td>\r\n                <td>");
            EndContext();
            BeginContext(1336, 18, false);
#line 65 "C:\Users\milja\source\repos\saga\saga\Views\Home\Users.cshtml"
               Write(Model[i].UserStat2);

#line default
#line hidden
            EndContext();
            BeginContext(1354, 27, true);
            WriteLiteral("</td>\r\n                <td>");
            EndContext();
            BeginContext(1382, 18, false);
#line 66 "C:\Users\milja\source\repos\saga\saga\Views\Home\Users.cshtml"
               Write(Model[i].UserStat3);

#line default
#line hidden
            EndContext();
            BeginContext(1400, 30, true);
            WriteLiteral("</td>\r\n\r\n\r\n            </tr>\r\n");
            EndContext();
#line 70 "C:\Users\milja\source\repos\saga\saga\Views\Home\Users.cshtml"
        }

#line default
#line hidden
            BeginContext(1441, 14, true);
            WriteLiteral("    </table>\r\n");
            EndContext();
#line 72 "C:\Users\milja\source\repos\saga\saga\Views\Home\Users.cshtml"
}
else
{

#line default
#line hidden
            BeginContext(1467, 33, true);
            WriteLiteral("    <h1>Data base is empty</h1>\r\n");
            EndContext();
#line 76 "C:\Users\milja\source\repos\saga\saga\Views\Home\Users.cshtml"
}

#line default
#line hidden
            BeginContext(1503, 2, true);
            WriteLiteral("\r\n");
            EndContext();
        }
        #pragma warning restore 1998
        [global::Microsoft.AspNetCore.Mvc.Razor.Internal.RazorInjectAttribute]
        public global::Microsoft.AspNetCore.Mvc.ViewFeatures.IModelExpressionProvider ModelExpressionProvider { get; private set; }
        [global::Microsoft.AspNetCore.Mvc.Razor.Internal.RazorInjectAttribute]
        public global::Microsoft.AspNetCore.Mvc.IUrlHelper Url { get; private set; }
        [global::Microsoft.AspNetCore.Mvc.Razor.Internal.RazorInjectAttribute]
        public global::Microsoft.AspNetCore.Mvc.IViewComponentHelper Component { get; private set; }
        [global::Microsoft.AspNetCore.Mvc.Razor.Internal.RazorInjectAttribute]
        public global::Microsoft.AspNetCore.Mvc.Rendering.IJsonHelper Json { get; private set; }
        [global::Microsoft.AspNetCore.Mvc.Razor.Internal.RazorInjectAttribute]
        public global::Microsoft.AspNetCore.Mvc.Rendering.IHtmlHelper<System.Collections.Generic.List<saga.Models.Consumer>> Html { get; private set; }
    }
}
#pragma warning restore 1591
