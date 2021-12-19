<%@ page import="java.sql.*" %>
<html>
<body>

<%
     String uname=request.getParameter("uname");
     String pwd=request.getParameter("pwd");
     try{
    	 Class.forName("com.mysql.cj.jdbc.Driver");
Connection con=DriverManager.getConnection("jdbc:mysql://localhost/csec","root","root");
Statement stmt=con.createStatement();
String q="SELECT * FROM user WHERE username='"+uname+"' AND password='"+pwd+"'";
ResultSet rs=stmt.executeQuery(q);
if(rs.next())
{
	out.println("Welcome");
}
else
{
	out.println("Credenial does,t exist in reocord");
}	
stmt.close();
con.close();
     }
     catch(Exception e)
     {
    	 out.println(e);
     }
 
%>


</body>
</html>