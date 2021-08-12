function timelength(mills) {
  
    const secs = mills / 1000;
    const fSecs = Math.floor(secs % 60);
    const min = secs / 60;
    const fMin = Math.floor(min % 60);
    const hours = min / 60;
    const fHours = Math.floor(hours % 24);
    const days = hours / 24;
    const fDays = Math.floor(days % 7);
    const weeks = Math.floor(days / 7);

    let hrsname = fHours == 1 ?"hour":"hours"
    let daysname = fDays == 1 ?"day":"days"
    let weeksname = weeks == 1 ?"wk":"wks"
    let minsname = fMin == 1 ?"min":"mins"
    let secsname = fSecs == 1 ?"sec":"secs"
    if (weeks > 0) {
      return `${weeks} ${weeksname}, ${fDays} ${daysname}`;
    } else if (fDays > 0) {
      return `${fDays} ${daysname}, ${fHours} ${hrsname}`;
    } else if (fHours > 0) {
      return `${fHours} ${hrsname}, ${fMin} ${minsname}`;
    } else if (fMin > 0) {
      return `${fMin} ${minsname}, ${fSecs} ${secsname}`;
    } else {
      return `${fSecs} ${secsname}`;
    }
}

function timedeltafordate(v) {
    
    const mills = (new Date()) - (new Date(v+"Z"));
    return timelength(mills)
}


function timedelta(v) {
    return timedeltafordate(new Date(v * 1000))

}

const uidsort = (a,b)=> {
  const uida= a.info.uid.split('-')[1]
  const uidb= b.info.uid.split('-')[1]
  return  parseInt(uidb) - parseInt(uida)
}

function timepretty(v) {
  return new Date(v * 1000).toLocaleString();
 
}
function formatNum(num, prec) {
    if(typeof num == 'undefined')
    return  null

  if (num == null) {
      return "";
  } else if (isNaN(num)) {
      return "NaN";
  } else {
      return num.toPrecision(prec);
  }
}

function numberToString(x) {
    if(typeof x == 'undefined')
        return "undefined"
    if( x == null)
        return "null"
    var parts = x.toString().split(".");
    parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    return parts.join(".");
}
function getImageId(uid) {
  let id = parseInt(uid.split("-")[1]);
  return id % 19;
}

export {getImageId, timedelta,timedeltafordate, timepretty, uidsort,timelength, formatNum, numberToString};
