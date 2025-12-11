class Paginator:

    @staticmethod
    def pagination(query,page,per_page):

            total = query.count()

            skip = (page - 1) * per_page
            data = query.offset(skip).limit(per_page).all()

            total_pages = (total + per_page - 1) // per_page

            return {
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": total_pages,
                "data": data
            }