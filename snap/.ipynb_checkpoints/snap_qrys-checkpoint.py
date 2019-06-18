qry0 = """SELECT 
	accountcoupon_couponusehistory.id  AS "accountcoupon_couponusehistory.id",
	branches.id  AS "branches.id",
	case when merchant_merchant.membership_zone_id = 1 then 'Melbourne'
    when merchant_merchant.membership_zone_id = 2 then 'Sydney'
    else null end AS "merchant_merchant.city",
	merchant_merchant.name  AS "merchant_merchant.name",
	branches.name  AS "branches.name"
FROM merchant_merchant  AS merchant_merchant
LEFT JOIN branch_branch  AS branches ON merchant_merchant.id = branches.merchant_id 
LEFT JOIN accountcoupon_couponusehistory  AS accountcoupon_couponusehistory ON branches.id =  accountcoupon_couponusehistory.branch_id

WHERE 
	(((accountcoupon_couponusehistory.time ) >= ((DATEADD('day', -364, CURRENT_DATE()))) AND (accountcoupon_couponusehistory.time ) < ((DATEADD('day', 365, DATEADD('day', -364, CURRENT_DATE()))))))
GROUP BY 1,2,3,4,5"""

qry1 = """
SELECT 
	merchant_merchant.name  AS "merchant_merchant.name",
	branches.name  AS "branches.name",
	branch_coordinates."LATITUDE"  AS "branch_coordinates.latitude",
	branch_coordinates."LONGITUDE"  AS "branch_coordinates.longitude",
	case when merchant_merchant.membership_zone_id = 1 then 'Melbourne'
    when merchant_merchant.membership_zone_id = 2 then 'Sydney'
    else null end AS "merchant_merchant.city",
	branches.id  AS "branches.id"
FROM merchant_merchant  AS merchant_merchant
LEFT JOIN branch_branch  AS branches ON merchant_merchant.id = branches.merchant_id 
LEFT JOIN BRANCH_COORDINATES  AS branch_coordinates ON branches.id = (branch_coordinates."ID") 

GROUP BY 1,2,3,4,5,6"""

qry2 = """SELECT 
	branches.id  AS "branches.id",
	case when merchant_merchant.membership_zone_id = 1 then 'Melbourne'
          when merchant_merchant.membership_zone_id = 2 then 'Sydney'
          else null end AS "merchant_merchant.city"
FROM merchant_merchant  AS merchant_merchant
LEFT JOIN branch_branch  AS branches ON merchant_merchant.id = branches.merchant_id 
LEFT JOIN accountcoupon_couponusehistory  AS accountcoupon_couponusehistory ON branches.id =  accountcoupon_couponusehistory.branch_id

WHERE 
	(((accountcoupon_couponusehistory.time ) >= ((DATEADD('day', -364, CURRENT_DATE()))) AND (accountcoupon_couponusehistory.time ) < ((DATEADD('day', 365, DATEADD('day', -364, CURRENT_DATE()))))))
GROUP BY 1,2"""