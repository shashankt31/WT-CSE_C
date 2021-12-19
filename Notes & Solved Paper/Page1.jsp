<html>
<body>
<%
String name="Bharath";
session.setAttribute("name",name);  //Storing data in session object
%>
<%= name %> , Session Started!!!
</body>
</html>