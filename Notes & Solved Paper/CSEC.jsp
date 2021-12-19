<html>
<body>
<%!
int room=2114;
public void printme()
{
	for(int i=0;i<5;i++)
	{
		%>
		"Hello"
		<%!
	}
}
%>


<%
printme();
String name=request.getParameter("name");
out.println("Welcome to class room no "+ room + name);
%>
</body>
</html>