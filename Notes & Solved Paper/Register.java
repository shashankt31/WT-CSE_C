import javax.servlet.http.*;
import java.io.*;
import java.sql.*;
public class Register extends HttpServlet
{

	public void service(HttpServletRequest req,HttpServletResponse res) throws IOException
	{
		String name=req.getParameter("name");
		String uname=req.getParameter("uname");
		String pass=req.getParameter("pass");
		
		res.setContentType("text/html");
		PrintWriter out=res.getWriter();
		
		try
		{
		Class.forName("com.mysql.cj.jdbc.Driver");
Connection con=DriverManager.getConnection("jdbc:mysql://localhost/cseasec","root","root");
Statement stmt=con.createStatement();
String q1="INSERT INTO user VALUES('"+name+"','"+uname+"','"+pass+"')";
int i=stmt.executeUpdate(q1);
if(i>0)
{
	out.println(name+" ,REGISTERED SUCCESFULLY!!");
}
			
			
			
		}
		catch(Exception e)
		{
			out.println(e);
		}
		
	}
}
