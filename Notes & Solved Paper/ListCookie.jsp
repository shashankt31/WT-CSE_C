<html>
<body>
<%
Cookie[] cookie=request.getCookies();
for(Cookie c:cookie)
{
	String name=c.getName();
	String value=c.getValue();
%>
<br/>Cookie Name:<%= name %> <br/> Cookie Value: <%= value %>
<%	
}
%>
</body>
</html>

HttpSession session=request.getSession()
session.setAttribute("name",)