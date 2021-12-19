<html>
<body>
<%
Cookie cookie1=new Cookie("name","Bhavani");
response.addCookie(cookie1);
%>
Cookie Created!!!
<form action="ListCookie.jsp">
<input type="submit" value="ListCookie"/>
</form>
</body>
</html>