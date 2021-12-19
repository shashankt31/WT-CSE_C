import javax.servlet.http.*;
import java.sql.*;
public class LoginServlet extends HttpServlet
{
   public void doPost(HttpServletRequest req,HttpServletResponse res)
   {
	   String uname=req.getParameter("uname");
	   String pwd=req.getParameter("pwd");
	   
	   res.setContentType("text/html");
	   PrintWriter out=res.getWriter();
	   
       try
       {
    	   
       }
       catch(Exception e)
       {
    	   out.printlne(e);
       }
	   
   }
	
}
