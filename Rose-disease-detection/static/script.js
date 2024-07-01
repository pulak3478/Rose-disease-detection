document.addEventListener("DOMContentLoaded", function () {
  // JavaScript code for sorting and manipulating list items
  var listItems = document.querySelectorAll("ol li");
  var sortedItems = Array.from(listItems).sort(function (a, b) {
      var aPercentage = parseFloat(a.textContent.split(":")[1]);
      var bPercentage = parseFloat(b.textContent.split(":")[1]);
      return bPercentage - aPercentage;
  });

  // Select the top 3 sorted items
  var top3Items = sortedItems.slice(0, 3);

  // Filter out items with a 0 value only if they are within the top 3 items
  top3Items = top3Items.filter(function(item) {
      return parseFloat(item.textContent.split(":")[1]) !== 0;
  });

  var ul = document.createElement("ol");
  top3Items.forEach(function (item) {
      ul.appendChild(item.cloneNode(true));
  });

  var originalUl = document.querySelector("ol");
  originalUl.innerHTML = ul.innerHTML;
});


// responsive

// Function to adjust the height of the iframe based on the viewport size
function adjustChatbotHeight() {
    var chatbotFrame = document.getElementById('chatbotFrame');
    var chatbotContainer = document.getElementById('cbot');

    chatbotFrame.style.height = (window.innerHeight * 0.95) + 'px';

}

adjustChatbotHeight();

window.addEventListener('resize', adjustChatbotHeight);
//bootstrap collabpsible

var coll = document.getElementsByClassName("collapsible");
var i;    
for (i = 0; i < coll.length; i++) {
  coll[i].addEventListener("click", function() {
    this.classList.toggle("active");
    var content = this.nextElementSibling;
    if (content.style.maxHeight){
      content.style.maxHeight = null;
    } else {
      content.style.maxHeight = content.scrollHeight + "px";
    } 
  });
}
//nav
function openNav() {
  document.getElementById("mySidebar").style.width = "150px";
  document.getElementById("main").style.marginLeft = "150px";
  document.getElementById("openbtn").style.visibility = "hidden"; // Hide the ☰ button
}

function closeNav() {
  document.getElementById("mySidebar").style.width = "0";
  document.getElementById("main").style.marginLeft= "0";
  document.getElementById("openbtn").style.visibility = "visible"; // Show the ☰ button
}