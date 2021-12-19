import javax.servlet.http.*;
import java.sql.*;
import java.io.*;
public class Login extends HttpServlet
{
	public void doPost(HttpServletRequest req,HttpServletResponse res) throws IOException
	{
		String uname=req.getParameter("uname");
		String pwd=req.getParameter("pwd");
		
		res.setContentType("text/html");
		PrintWriter out=res.getWriter();
		try {
		
Class.forName("com.mysql.cj.jdbc.Driver");
Connection c=DriverManager.getConnection("jdbc:mysql://localhost/cseasec","root","root");
Statement stmt=c.createStatement();
String q="SELECT * FROM user WHERE username='"+uname+"' AND password='"+pwd+"'";
ResultSet rs=stmt.executeQuery(q);
if(rs.next())
{
	out.println("Welcome"+rs.getString("name")+" to my website");
	return;
}
String q2="SELECT * FROM user WHERE username='"+uname+"' AND password!='"+pwd+"'";
ResultSet rs1=stmt.executeQuery(q2);
if(rs1.next())
{
	out.println("INCORRECT PASSWORD!!");
	return;
}
else
{
	res.sendRedirect("http://localhost:8080/Sample0/register.html");
}
		}


		catch(Exception e)
		{
			out.println(e);
		}
		
	}
}